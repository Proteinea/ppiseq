import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from ppi_research import data_adapters
from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import EmbedConcatConvBERTModel
from ppi_research.utils import create_run_name
from ppi_research.utils import set_seed
from transformers import T5EncoderModel
from transformers import T5Tokenizer
from transformers import Trainer
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels
from ppi_research.utils import get_default_training_args


@hydra.main(
    config_path="../../config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.prott5.ckpt
    max_length = cfg.prott5.max_length
    seed = cfg.train_config.seed
    set_seed(seed=seed)
    print("Checkpoint:", ckpt)
    tokenizer = T5Tokenizer.from_pretrained(ckpt)
    model = T5EncoderModel.from_pretrained(ckpt)

    downstream_model = EmbedConcatConvBERTModel(
        backbone=model,
        pooler=cfg.pooler,
        concat_first=cfg.embed_concat_config.concat_first,
        model_name="prott5",
        embedding_name="last_hidden_state",
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="convbert_embed_concat",
        pooler=cfg.pooler,
        concat_first=cfg.embed_concat_config.concat_first,
        seed=seed,
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
