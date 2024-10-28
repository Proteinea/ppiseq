import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_PROJECT'] = 'PPIExperiments'
# os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import torch
from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer, EvalPrediction, AutoTokenizer
from torch import nn
import torch
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
import random
import numpy as np
from protbench import metrics

seed = 7
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

def create_run_name(**kwargs) -> str:
    output = ""
    for k, v in kwargs.items():
        if isinstance(v, list):
            v = "_".join(v)
        output += f"{k}_{v}-"
    return output[:-1]


def global_mean_pooling1d(x, padding_mask=None):
    if padding_mask is None:
        return torch.mean(x, dim=1)

    x_masked = x * padding_mask.unsqueeze(-1)
    return x_masked.sum(1) / padding_mask.sum(1)



class Perceiver(nn.Module):
    def __init__(self, embed_dim, num_latents, num_heads=8, attn_dropout=0.0, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.bias = bias
        self.latents = nn.Parameter(torch.ones((num_latents, embed_dim)))

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=bias,
            batch_first=True,
        )

    def forward(self, inputs, attention_mask=None):
        batch_size = inputs.shape[0]
        latents = self.latents.unsqueeze(dim=0)
        latents = torch.repeat_interleave(latents, batch_size, dim=0)

        attention_mask = attention_mask.to(
            dtype=inputs.dtype,
            device=inputs.device,
        )

        # attention_mask: [B, T]
        # latents_attn_mask: [B, T]
        latents_attn_mask = torch.ones(
            (batch_size, self.num_latents),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((latents_attn_mask, attention_mask), dim=-1)
        inputs = torch.cat((latents, inputs), dim=1)

        outputs, _ = self.attn(
            query=latents,
            key=inputs,
            value=inputs,
            key_padding_mask=attention_mask.log(),
            need_weights=False,
        )
        return outputs


class PPIAttnLoRA(nn.Module):
    def __init__(
        self,
        backbone,
        num_latents=512,
        num_heads=8,
        attn_dropout=0.0,
        bias=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.config.hidden_size
        self.output = nn.Linear(self.embed_dim, 1)
        self.perceiver = Perceiver(
            embed_dim=self.embed_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(self, protein_1, protein_2, attention_mask_1=None, attention_mask_2=None, labels=None):
        protein_1_embed = self.backbone(input_ids=protein_1, attention_mask=attention_mask_1)[0]
        protein_2_embed = self.backbone(input_ids=protein_2, attention_mask=attention_mask_2)[0]
        output_1 = self.perceiver(inputs=protein_1_embed, attention_mask=attention_mask_1)
        output_2 = self.perceiver(inputs=protein_2_embed, attention_mask=attention_mask_2)
        output = output_1 + output_2
        pooled_output = global_mean_pooling1d(output)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)
        
        return {
            "logits": logits,
            "loss": loss,
        }


def compute_ppi_metrics(p: EvalPrediction):
    spearman_stat = metrics.compute_spearman(p)
    num_examples = p.label_ids.shape[0]
    error_bar = metrics.compute_error_bar_for_regression(
        spearman_corr=spearman_stat, num_examples=num_examples
    )
    rmse = metrics.compute_rmse(p)
    pearson_corr = metrics.compute_pearsonr(p)
    return {
        "spearman": spearman_stat,
        "error_bar": error_bar,
        "pearsonr": pearson_corr,
        "rmse": rmse,
    }


class Dataset:
    def __init__(self, hf_ds):
        self.hf_ds = hf_ds


    def __len__(self):
        return self.hf_ds.num_rows

    
    def __getitem__(self, idx):
        return {
            "affinity": self.hf_ds['affinity (pKd)'][idx],
            "protein_1": self.hf_ds['protein 1 sequence'][idx],
            "protein_2": self.hf_ds['protein 2 sequence'][idx],
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        seqs_1, seqs_2, labels = [], [], []
        for b in batch:
            seqs_1.append(b['protein_1'])
            seqs_2.append(b['protein_2'])
            labels.append(b['affinity'])
        
        seqs_1_encoded = self.tokenizer(
            seqs_1,
            add_special_tokens=True,
            max_length=None,
            padding=True,
            truncation=False,
            return_tensors='pt',
        )
        seqs_2_encoded = self.tokenizer(
            seqs_2,
            add_special_tokens=True,
            max_length=None,
            padding=True,
            truncation=False,
            return_tensors='pt',
        )
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "protein_1": seqs_1_encoded['input_ids'],
            "protein_2": seqs_2_encoded['input_ids'],
            "attention_mask_1": seqs_1_encoded['attention_mask'],
            "attention_mask_2": seqs_2_encoded['attention_mask'],
            "labels": labels,
        }


def main():
    ckpt = "ElnaggarLab/ankh-large"
    print("Checkpoint: ", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)
    r = 16
    alpha = 32
    target_modules = ["q", "v"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias='none',
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, lora_config).encoder
    downstream_model = PPIAttnLoRA(peft_model)

    run_name = create_run_name(
        backbone=ckpt,
        setup="lora_perceiver",
        r=r,
        alpha=alpha,
        target_modules=target_modules,
    )

    training_args = TrainingArguments(
        output_dir=run_name + "_weights",
        run_name=run_name,
        num_train_epochs=20,
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,
        warmup_steps=1000,
        learning_rate=1e-3,
        weight_decay=0.0,
        logging_dir=f"./logs_{run_name}",
        logging_steps=1,
        do_train=True,
        do_eval=True,
        eval_strategy='epoch',
        gradient_accumulation_steps=16,
        fp16=False,
        fp16_opt_level="02",
        seed=seed,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model='eval_validation_spearman',
        greater_is_better=True,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        save_safetensors=False,
    )

    ds = load_dataset("proteinea/skempi_ppi")
    train_ds = Dataset(ds['train'])
    val_ds = Dataset(ds['validation'])
    test_ds = Dataset(ds['test'])

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        data_collator=Collator(tokenizer=tokenizer),
        train_dataset=train_ds,
        eval_dataset={'validation': val_ds, "test": test_ds},
        compute_metrics=compute_ppi_metrics
    )

    trainer.train()


if __name__ == "__main__":
    main()
    


