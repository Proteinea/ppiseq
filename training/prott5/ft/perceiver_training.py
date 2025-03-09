import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from ppi_research import data_adapters
from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import PerceiverModel
from ppi_research.training_utils import create_run_name
from ppi_research.training_utils import add_lora_prefix
from ppi_research.training_utils import set_seed
from transformers import Trainer
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels
from ppi_research.training_utils import get_default_training_args
from ppi_research.models.backbones import load_prott5_model


@hydra.main(
    config_path="../../config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.prott5.ckpt
    max_length = cfg.prott5.max_length
    use_lora = cfg.use_lora
    seed = cfg.train_config.seed
    set_seed(seed=seed)
    print("Checkpoint:", ckpt)

    model, tokenizer = load_prott5_model(
        ckpt=ckpt,
        use_lora=use_lora,
        rank=cfg.lora_config.r,
        alpha=cfg.lora_config.alpha,
        dropout=cfg.lora_config.dropout,
        target_modules=cfg.prott5.target_modules,
        bias=cfg.lora_config.bias,
    )

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
        gradient_checkpointing=cfg.enable_gradient_checkpointing,
        loss_fn=cfg.loss_config.name,
        loss_fn_options=cfg.loss_config.options,
    )

    setup = add_lora_prefix("perceiver", use_lora=use_lora)
    run_name = create_run_name(
        backbone=ckpt,
        setup=setup,
        r=cfg.lora_config.r,
        num_latents=cfg.perceiver_config.num_latents,
        alpha=cfg.lora_config.alpha,
        target_modules=cfg.prott5.target_modules,
        pooler=cfg.pooler,
        seed=seed,
        loss_fn=cfg.loss_config.name,
    )

    training_args = get_default_training_args(
        run_name,
        seed,
        **cfg.train_config,
    )

    train_ds, eval_datasets = ppi_datasets.load_ppi_dataset(
        cfg.dataset_config.repo_id,
        cfg.dataset_config.name,
    )

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
