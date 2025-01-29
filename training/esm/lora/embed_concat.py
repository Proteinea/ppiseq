import os

from ppi_research.layers import poolers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
# os.environ['WANDB_MODE'] = 'disabled'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


from peft import LoraConfig
from peft import get_peft_model
from ppi_research import data_adapters
from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import EmbedConcatModel
from ppi_research.utils import create_run_name
from ppi_research.utils import set_seed
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.esm.ckpt
    max_length = cfg.dataset_config.max_length
    seed = cfg.train_config.seed
    set_seed(seed=seed)
    print("Checkpoint:", ckpt)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)

    lora_config = LoraConfig(
        r=cfg.esm.lora.r,
        lora_alpha=cfg.esm.lora.alpha,
        lora_dropout=cfg.esm.lora.dropout,
        bias=cfg.esm.lora.bias,
        target_modules=cfg.esm.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)

    pooler = poolers.get(
        cfg.downstream_config.pooler,
        embed_dim=model.config.hidden_size,
    )
    downstream_model = EmbedConcatModel(
        backbone=model,
        pooler=pooler,
        model_name="esm2",
        embedding_name="last_hidden_state",
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="lora_embed_concat",
        r=cfg.esm.lora.r,
        alpha=cfg.esm.lora.alpha,
        target_modules=cfg.esm.lora.target_modules,
        pooler=cfg.downstream_config.pooler,
        seed=seed,
    )

    training_args = TrainingArguments(
        output_dir=run_name + "_weights",
        run_name=run_name,
        num_train_epochs=30,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=1000,
        learning_rate=5e-4,
        weight_decay=0.0,
        logging_dir=f"./logs_{run_name}",
        logging_steps=1,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        gradient_accumulation_steps=32,
        fp16=False,
        fp16_opt_level="02",
        seed=seed,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_validation_rmse",
        greater_is_better=False,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        save_safetensors=False,
    )

    train_ds, eval_datasets = ppi_datasets.load_ppi_dataset(
        cfg.dataset_config.dataset_name,
    )

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        data_collator=data_adapters.PairCollator(
            tokenizer=tokenizer, max_length=max_length, random_swapping=True
        ),
        train_dataset=train_ds,
        eval_dataset=eval_datasets,
        compute_metrics=compute_ppi_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
