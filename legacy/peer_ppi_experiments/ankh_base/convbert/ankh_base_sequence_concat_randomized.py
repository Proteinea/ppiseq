import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_PROJECT'] = 'PEERPPIConvBERTExperiments'
# os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

import torch
from transformers import T5EncoderModel, TrainingArguments, Trainer, EvalPrediction, AutoTokenizer
from torch import nn
import torch
from datasets import load_dataset
import random
import numpy as np
from protbench import metrics
from transformers.models import convbert

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


class PPIConvBERT(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.config.hidden_size

        convbert_config = convbert.ConvBertConfig(
            hidden_size=self.embed_dim,
            num_hidden_layers=1,
            num_attention_heads=8,
            intermediate_size=self.embed_dim // 2,
            conv_kernel_size=7,
        )
    
        # We use only one convbert layer in
        # our benchmarking so we just use `ConvBertLayer`.
        self.convbert_layer = convbert.ConvBertLayer(convbert_config)

        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def _extract_embeddings(self, input_ids, attention_mask=None):
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
        return outputs

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed = self._extract_embeddings(input_ids, attention_mask)
        embed = self.convbert_layer(embed)[0]

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
            "affinity": self.hf_ds['interaction'][idx],
            "protein_1": self.hf_ds['graph1'][idx],
            "protein_2": self.hf_ds['graph2'][idx],
        }


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        sequences, labels = [], []

        for b in batch:
            if random.random() < 0.5:
                inputs = (b['protein_1'], b['protein_2'])
            else:
                inputs = (b['protein_2'], b['protein_1'])
            sequences.append(inputs)
            labels.append(b['affinity'])

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
    model = T5EncoderModel.from_pretrained(ckpt)
    downstream_model = PPIConvBERT(model)

    run_name = create_run_name(
        backbone=ckpt,
        setup="convbert_sequence_concat_randomized",
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

    data_files = {
        "train": "train_split.csv",
        "validation": "valid_split.csv",
        "test": "test_split.csv",
    }
    ds = load_dataset("proteinea/peer_ppi_splits", data_files=data_files)
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
    


