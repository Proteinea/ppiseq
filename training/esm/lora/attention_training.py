import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
# os.environ['WANDB_MODE'] = 'disabled'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from transformers import AutoTokenizer
from transformers import AutoModel
from peft import LoraConfig
from peft import get_peft_model
from ppi_research.data_adapters import ppi_datasets
from ppi_research.utils import create_run_name
from ppi_research.models import AttnPoolAddModel
from transformers import Trainer
from transformers import TrainingArguments
from ppi_research import data_adapters
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.utils import set_seed
from ppi_research.utils import esm_checkpoint_mapping
from ppi_research.utils import esm_checkpoints
import argparse


seed = 7
set_seed(seed=seed)


def main(args):
    ckpt = args.ckpt
    ds_name = args.ds_name
    max_length = args.max_length
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)
    r = 16
    alpha = 32
    target_modules = ["query", "value"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
    )

    peft_model = get_peft_model(model, lora_config)
    downstream_model = AttnPoolAddModel(peft_model)

    run_name = create_run_name(
        backbone=ckpt,
        setup="lora_attn_pooled_addition",
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
        eval_strategy="epoch",
        gradient_accumulation_steps=16,
        fp16=False,
        fp16_opt_level="02",
        seed=seed,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_validation_spearman",
        greater_is_better=True,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        save_safetensors=False,
    )

    train_ds, eval_datasets = ppi_datasets.load_ppi_dataset(ds_name)

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        data_collator=data_adapters.PairCollator(
            tokenizer=tokenizer, max_length=max_length
        ),
        train_dataset=train_ds,
        eval_dataset=eval_datasets,
        compute_metrics=compute_ppi_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        choices=esm_checkpoints(),
    )
    argparser.add_argument(
        "--ds_name",
        type=str,
        required=True,
        choices=list(ppi_datasets.available_datasets.keys()),
    )
    argparser.add_argument(
        "--max_length",
        type=int,
        default=None,
        required=False,
    )
    args = argparser.parse_args()
    args.ckpt = esm_checkpoint_mapping(args.ckpt)
    main(args)
