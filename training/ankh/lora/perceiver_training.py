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
from ppi_research.models import PerceiverModel
from ppi_research.utils import create_run_name
from ppi_research.utils import set_seed
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
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
    ckpt = cfg.ankh.ckpt
    max_length = cfg.max_length
    seed = cfg.train_config.seed
    print("Checkpoint:", ckpt)
    set_seed(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)

    lora_config = LoraConfig(
        r=cfg.lora_config.r,
        lora_alpha=cfg.lora_config.alpha,
        lora_dropout=cfg.lora_config.dropout,
        bias=cfg.lora_config.bias,
        target_modules=cfg.ankh.target_modules,
    )

    model = get_peft_model(model, lora_config).encoder
    pooler = poolers.get(
        cfg.downstream_config.pooler,
        embed_dim=model.config.hidden_size,
    )
    downstream_model = PerceiverModel(
        backbone=model,
        pooler=pooler,
        model_name="ankh",
        embedding_name="last_hidden_state",
        num_latents=cfg.perceiver_config.num_latents,
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="lora_perceiver",
        r=cfg.lora_config.r,
        num_latents=cfg.perceiver_config.num_latents,
        alpha=cfg.lora_config.alpha,
        target_modules=cfg.ankh.target_modules,
        pooler=cfg.downstream_config.pooler,
        seed=seed,
    )

    training_args = TrainingArguments(
        output_dir=run_name + "_weights",
        run_name=run_name,
        num_train_epochs=cfg.train_config.num_train_epochs,
        per_device_train_batch_size=cfg.train_config.per_device_train_batch_size, # noqa
        per_device_eval_batch_size=cfg.train_config.per_device_eval_batch_size,
        warmup_steps=cfg.train_config.warmup_steps,
        learning_rate=cfg.train_config.learning_rate,
        weight_decay=cfg.train_config.weight_decay,
        logging_dir=f"./logs_{run_name}",
        logging_steps=cfg.train_config.logging_steps,
        do_train=True,
        do_eval=True,
        eval_strategy=cfg.train_config.eval_strategy,
        gradient_accumulation_steps=cfg.train_config.gradient_accumulation_steps,  # noqa
        fp16=False,
        fp16_opt_level="02",
        seed=seed,
        load_best_model_at_end=cfg.train_config.load_best_model_at_end,
        save_total_limit=cfg.train_config.save_total_limit,
        metric_for_best_model=cfg.train_config.metric_for_best_model,
        greater_is_better=cfg.train_config.greater_is_better,
        save_strategy=cfg.train_config.save_strategy,
        report_to="wandb",
        remove_unused_columns=cfg.train_config.remove_unused_columns,
        save_safetensors=cfg.train_config.save_safetensors,
    )

    train_ds, eval_datasets = ppi_datasets.load_ppi_dataset(
        cfg.dataset_config.dataset_name,
    )

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
    main()
