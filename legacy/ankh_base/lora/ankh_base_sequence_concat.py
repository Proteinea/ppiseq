import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_PROJECT'] = 'PPIRefExperiments'
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


class PPIAttnLoRA(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.config.hidden_size
        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]

        attention_mask = attention_mask.to(
            device=embed.device,
            dtype=embed.dtype,
        )
        pooled_output = global_mean_pooling1d(embed, attention_mask)
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
        sequences, labels = [], []

        for b in batch:
            sequences.append((b['protein_1'], b['protein_2']))
            labels.append(b['affinity'])


        # Seq1 </s> Seq2 </s>
        encoded_sequences = self.tokenizer(
            sequences,
            add_special_tokens=True,
            max_length=None,
            padding=True,
            truncation=False,
            return_tensors='pt',
        )
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "input_ids": encoded_sequences["input_ids"],
            "attention_mask": encoded_sequences["attention_mask"],
            "labels": labels,
        }


def main():
    ckpt = "ElnaggarLab/ankh-base"
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
        setup="lora_sequence_concat",
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
    


