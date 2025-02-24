from functools import partial
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "PPIRefExperiments"
os.environ['WANDB_MODE'] = 'disabled'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from ppi_research import data_adapters
from ppi_research.data_adapters import ppi_datasets
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.models import MultiChainConvBERTModel
from ppi_research.utils import create_run_name
from ppi_research.utils import get_default_training_args
from ppi_research.utils import set_seed
from transformers import T5EncoderModel
from transformers import AutoTokenizer
from transformers import Trainer
import hydra
from omegaconf import DictConfig
from ppi_research.data_adapters.preprocessing import log_transform_labels


@hydra.main(
    config_path="../../config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    ckpt = cfg.ankh.ckpt
    max_length = cfg.ankh.max_length
    seed = cfg.train_config.seed

    set_seed(seed=seed)
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = T5EncoderModel.from_pretrained(ckpt)

    downstream_model = MultiChainConvBERTModel(
        backbone=model,
        global_pooler=cfg.multichain_config.global_pooler,
        chains_pooler=cfg.multichain_config.chains_pooler,
        shared_global_pooler=cfg.multichain_config.shared_global_pooler,
        shared_chains_pooler=cfg.multichain_config.shared_chains_pooler,
        shared_convbert=cfg.multichain_config.shared_convbert,
        aggregation_method=cfg.multichain_config.aggregation_method,
        use_ffn=cfg.multichain_config.use_ffn,
        bias=cfg.multichain_config.bias,
        model_name="ankh",
        embedding_name="last_hidden_state",
    )

    run_name = create_run_name(
        backbone=ckpt,
        setup="convbert_multichain",
        seed=seed,
        shared_convbert=cfg.multichain_config.shared_convbert,
        aggregation_method=cfg.multichain_config.aggregation_method,
        use_ffn=cfg.multichain_config.use_ffn,
        bias=cfg.multichain_config.bias,
        global_pooler=cfg.multichain_config.global_pooler,
        chains_pooler=cfg.multichain_config.chains_pooler,
        shared_global_pooler=cfg.multichain_config.shared_global_pooler,
        shared_chains_pooler=cfg.multichain_config.shared_chains_pooler,
    )

    training_args = get_default_training_args(
        run_name,
        seed,
        **cfg.train_config,
    )

    train_ds, eval_datasets = ppi_datasets.load_ppi_dataset(cfg.dataset_name)

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        data_collator=data_adapters.MultiChainCollator(
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
