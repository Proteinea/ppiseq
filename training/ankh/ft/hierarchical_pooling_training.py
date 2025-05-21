import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hydra
from omegaconf import DictConfig
from ppiseq import data_adapters
from ppiseq.data_adapters import ppi_datasets
from ppiseq.data_adapters.preprocessing import log_transform_labels
from ppiseq.metrics import compute_ppi_metrics
from ppiseq.models import HierarchicalPoolingModel
from ppiseq.models.backbones import load_ankh_model
from ppiseq.training_utils import add_lora_prefix
from ppiseq.training_utils import create_run_name
from ppiseq.training_utils import get_default_training_args
from ppiseq.training_utils import set_seed
from transformers import Trainer


@hydra.main(
    config_path="../../config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.ankh.ckpt
    max_length = cfg.ankh.max_length
    seed = cfg.train_config.seed
    use_lora = cfg.use_lora

    set_seed(seed=seed)
    print("Checkpoint:", ckpt)

    model, tokenizer = load_ankh_model(
        ckpt=ckpt,
        use_lora=use_lora,
        rank=cfg.lora_config.r,
        alpha=cfg.lora_config.alpha,
        dropout=cfg.lora_config.dropout,
        target_modules=cfg.ankh.target_modules,
        bias=cfg.lora_config.bias,
    )

    downstream_model = HierarchicalPoolingModel(
        backbone=model,
        global_pooler=cfg.hierarchical_pooling_config.global_pooler,
        chains_pooler=cfg.hierarchical_pooling_config.chains_pooler,
        shared_global_pooler=cfg.hierarchical_pooling_config.shared_global_pooler, # noqa
        shared_chains_pooler=cfg.hierarchical_pooling_config.shared_chains_pooler, # noqa
        aggregation_method=cfg.hierarchical_pooling_config.aggregation_method,
        use_ffn=cfg.hierarchical_pooling_config.use_ffn,
        bias=cfg.hierarchical_pooling_config.bias,
        model_name="ankh",
        embedding_name="last_hidden_state",
        gradient_checkpointing=cfg.enable_gradient_checkpointing,
        loss_fn=cfg.loss_config.name,
        loss_fn_options=cfg.loss_config.options,
    )

    setup = add_lora_prefix("hierarchical_pooling", use_lora=use_lora)
    run_name = create_run_name(
        backbone=ckpt,
        setup=setup,
        seed=seed,
        aggregation_method=cfg.hierarchical_pooling_config.aggregation_method,
        use_ffn=cfg.hierarchical_pooling_config.use_ffn,
        bias=cfg.hierarchical_pooling_config.bias,
        global_pooler=cfg.hierarchical_pooling_config.global_pooler,
        chains_pooler=cfg.hierarchical_pooling_config.chains_pooler,
        shared_global_pooler=cfg.hierarchical_pooling_config.shared_global_pooler, # noqa
        shared_chains_pooler=cfg.hierarchical_pooling_config.shared_chains_pooler, # noqa
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
        data_collator=data_adapters.HierarchicalPoolingCollator(
            tokenizer=tokenizer,
            model_name="ankh",
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
