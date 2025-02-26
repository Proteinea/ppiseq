import os
import functools
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from peft import LoraConfig
from peft import get_peft_model
from ppi_research.data_adapters import ppi_datasets
from ppi_research.data_adapters.collators import SequenceConcatCollator
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import SequenceConcatModel
from ppi_research.utils import create_run_name
from ppi_research.utils import set_seed
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import Trainer
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels
from ppi_research.utils import get_default_training_args


@hydra.main(
    config_path="../../config",
    version_base=None,
    config_name="train_config",
)
def main(cfg: DictConfig):
    ckpt = cfg.prott5.ckpt
    max_length = cfg.prott5.max_length
    seed = cfg.train_config.seed
    set_seed(seed=seed)
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

    downstream_model = SequenceConcatModel(
        backbone=model,
        pooler=cfg.pooler,
        model_name="prott5",
        embedding_name="last_hidden_state",
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="lora_sequence_concat",
        r=cfg.lora_config.r,
        alpha=cfg.lora_config.alpha,
        target_modules=cfg.prott5.target_modules,
        pooler=cfg.pooler,
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
        data_collator=SequenceConcatCollator(
            tokenizer=tokenizer,
            model_name="prott5",
            max_length=max_length,
            labels_preprocessing_function=functools.partial(
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
