import os
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


from ppi_research.data_adapters import ppi_datasets
from ppi_research.data_adapters.collators import SequenceConcatCollator
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import SequenceConcatConvBERTModel
from ppi_research.utils import create_run_name
from ppi_research.utils import get_default_training_args
from ppi_research.utils import set_seed
from transformers import AutoTokenizer
from transformers import T5EncoderModel
from transformers import Trainer
from ppi_research.data_adapters.preprocessing import log_transform_labels
import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../config", config_name="train_config", version_base=None
)
def main(cfg: DictConfig):
    ckpt = cfg.ankh.ckpt
    max_length = cfg.ankh.max_length
    seed = cfg.train_config.seed
    set_seed(seed=seed)
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = T5EncoderModel.from_pretrained(ckpt)

    downstream_model = SequenceConcatConvBERTModel(
        backbone=model,
        pooler=cfg.pooler,
        model_name="ankh",
        embedding_name="last_hidden_state",
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="convbert_sequence_concat",
        seed=seed,
        pooler=cfg.pooler,
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
        data_collator=SequenceConcatCollator(
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
