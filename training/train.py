import os
from functools import partial

os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.training_utils import get_ppi_model
from ppi_research.training_utils import create_run_name
from ppi_research.training_utils import set_seed
from ppi_research.training_utils import get_collator_cls
from transformers import Trainer
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels
from ppi_research.training_utils import get_default_training_args
from ppi_research.training_utils import validate_config
from ppi_research.training_utils import get_model_name_from_ckpt
from ppi_research.models.backbones import load_backbone
from transformers.trainer_callback import EarlyStoppingCallback


@hydra.main(
    config_path="config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    validate_config(cfg)
    ckpt = cfg.ckpt
    model_name = get_model_name_from_ckpt(ckpt)
    seed = cfg.train_config.seed
    set_seed(seed=seed)
    print("Checkpoint:", ckpt)

    model, tokenizer = load_backbone(
        ckpt=ckpt,
        use_lora=cfg.lora_config.enable,
        rank=cfg.lora_config.r,
        alpha=cfg.lora_config.alpha,
        dropout=cfg.lora_config.dropout,
        target_modules=cfg.lora_config.target_modules,
        bias=cfg.lora_config.bias,
    )

    downstream_model = get_ppi_model(
        backbone=model, model_name=model_name, cfg=cfg
    )
    print(downstream_model)
    run_name = create_run_name(cfg)

    training_args = get_default_training_args(run_name, **cfg.train_config)

    train_ds, eval_datasets = ppi_datasets.load_ppi_dataset(
        cfg.dataset_config.repo_id,
        cfg.dataset_config.name,
    )

    collator = get_collator_cls(cfg.architecture)

    callbacks = []
    if cfg.early_stop_config.enable:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stop_config.patience,
                early_stopping_threshold=cfg.early_stop_config.threshold,
            )
        )

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        data_collator=collator(
            tokenizer=tokenizer,
            model_name=model_name,
            max_length=cfg.max_length,
            labels_preprocessing_function=partial(
                log_transform_labels,
                base=cfg.label_transform_config.log_base,
                eps=cfg.label_transform_config.eps,
            ),
        ),
        train_dataset=train_ds,
        eval_dataset=eval_datasets,
        compute_metrics=compute_ppi_metrics,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    main()
