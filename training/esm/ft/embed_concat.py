import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from ppi_research import data_adapters
from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import EmbedConcatModel
from ppi_research.models.backbones import load_esm_model
from ppi_research.utils import create_run_name
from ppi_research.utils import set_seed
from transformers import Trainer
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels
from ppi_research.utils import get_default_training_args
from ppi_research.utils import add_lora_prefix


@hydra.main(
    config_path="../../config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.esm.ckpt
    max_length = cfg.esm.max_length
    seed = cfg.train_config.seed
    use_lora = cfg.use_lora
    set_seed(seed=seed)
    print("Checkpoint:", ckpt)

    model, tokenizer = load_esm_model(
        ckpt=ckpt,
        use_lora=use_lora,
        rank=cfg.lora_config.r,
        alpha=cfg.lora_config.alpha,
        dropout=cfg.lora_config.dropout,
        target_modules=cfg.esm.target_modules,
        bias=cfg.lora_config.bias,
    )

    downstream_model = EmbedConcatModel(
        backbone=model,
        pooler=cfg.pooler,
        concat_first=cfg.embed_concat_config.concat_first,
        model_name="esm2",
        embedding_name="last_hidden_state",
        gradient_checkpointing=cfg.enable_gradient_checkpointing,
        loss_fn=cfg.loss_config.name,
        loss_fn_options=cfg.loss_config.options,
    )

    setup = add_lora_prefix("embed_concat", use_lora=use_lora)
    run_name = create_run_name(
        backbone=ckpt,
        setup=setup,
        r=cfg.lora_config.r,
        alpha=cfg.lora_config.alpha,
        target_modules=cfg.esm.target_modules,
        pooler=cfg.pooler,
        concat_first=cfg.embed_concat_config.concat_first,
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
            model_name="esm",
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
