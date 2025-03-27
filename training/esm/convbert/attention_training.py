import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from ppi_research import data_adapters
from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import AttnPoolAddConvBERTModel
from ppi_research.training_utils import create_run_name
from ppi_research.training_utils import set_seed
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import Trainer
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels
from ppi_research.training_utils import get_default_training_args


@hydra.main(
    config_path="../../config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.esm.ckpt
    max_length = cfg.esm.max_length
    seed = cfg.train_config.seed

    set_seed(seed=seed)
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)

    downstream_model = AttnPoolAddConvBERTModel(
        backbone=model,
        pooler=cfg.pooler,
        shared_convbert=cfg.attn_pool_add_config.shared_convbert,
        shared_attention=cfg.attn_pool_add_config.shared_attention,
        convbert_dropout=cfg.convbert_config.convbert_dropout,
        convbert_attn_dropout=cfg.convbert_config.convbert_attn_dropout,
        use_ffn=cfg.attn_pool_add_config.use_ffn,
        ffn_multiplier=cfg.attn_pool_add_config.ffn_multiplier,
        model_name="esm2",
        embedding_name="last_hidden_state",
        loss_fn=cfg.loss_config.name,
        loss_fn_options=cfg.loss_config.options,
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="convbert_attn_pool_add",
        pooler=cfg.pooler,
        seed=seed,
        shared_convbert=cfg.attn_pool_add_config.shared_convbert,
        shared_attention=cfg.attn_pool_add_config.shared_attention,
        use_ffn=cfg.attn_pool_add_config.use_ffn,
        ffn_multiplier=cfg.attn_pool_add_config.ffn_multiplier,
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
