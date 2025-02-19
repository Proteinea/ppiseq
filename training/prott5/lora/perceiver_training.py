import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ['WANDB_MODE'] = 'disabled'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from peft import LoraConfig
from peft import get_peft_model
from ppi_research import data_adapters
from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import PerceiverModel
from ppi_research.utils import create_run_name
from ppi_research.utils import set_seed
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import Trainer
from transformers import TrainingArguments
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels

seed = 7
set_seed(seed=seed)


@hydra.main(
    config_path="../../config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.prott5.ckpt
    max_length = cfg.prott5.max_length
    print("Checkpoint:", ckpt)

    tokenizer = T5Tokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)
    lora_config = LoraConfig(
        r=cfg.lora_config.r,
        lora_alpha=cfg.lora_config.alpha,
        lora_dropout=cfg.lora_config.dropout,
        bias=cfg.lora_config.bias,
        target_modules=cfg.prott5.target_modules,
    )

    model = get_peft_model(model, lora_config).encoder

    downstream_model = PerceiverModel(
        backbone=model,
        pooler=cfg.pooler,
        model_name="prott5",
        embedding_name="last_hidden_state",
        num_latents=cfg.perceiver_config.num_latents,
        num_heads=cfg.perceiver_config.num_heads,
        hidden_dim=cfg.perceiver_config.hidden_dim,
        bias=cfg.perceiver_config.bias,
        num_perceiver_layers=cfg.perceiver_config.num_perceiver_layers,
        num_self_layers=cfg.perceiver_config.num_self_layers,
        activation=cfg.perceiver_config.activation,
        gated=cfg.perceiver_config.gated,
        shared_perceiver=cfg.perceiver_config.shared_perceiver,
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="lora_perceiver",
        r=cfg.lora_config.r,
        num_latents=cfg.perceiver_config.num_latents,
        alpha=cfg.lora_config.alpha,
        target_modules=cfg.prott5.target_modules,
        pooler=cfg.pooler,
        seed=seed,
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
        metric_for_best_model="eval_validation_rmse",
        greater_is_better=False,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        save_safetensors=False,
    )

    train_ds, eval_datasets = ppi_datasets.load_ppi_dataset(cfg.dataset_name)

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        data_collator=data_adapters.PairCollator(
            tokenizer=tokenizer,
            model_name="prott5",
            max_length=max_length,
            labels_preprocessing_function=partial(
                log_transform_labels,
                base=cfg.label_transform_config.log_base,
                eps=cfg.label_transform_config.eps,
            ),
        ),
        train_dataset=train_ds,
        eval_dataset=eval_datasets,
        compute_metrics=compute_ppi_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
