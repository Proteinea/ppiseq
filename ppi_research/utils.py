import random

import numpy as np
import torch
from transformers import TrainingArguments


def create_run_name(**kwargs) -> str:
    output = ""
    for k, v in kwargs.items():
        if isinstance(v, list):
            v = "_".join(v)
        output += f"{k}_{v}-"
    return output[:-1]


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def get_default_training_args(run_name: str, seed: int, **train_config):
    train_config.pop("output_dir", None)
    train_config.pop("run_name", None)
    train_config.pop("logging_dir", None)
    train_config.pop("do_train", None)
    train_config.pop("do_eval", None)
    train_config.pop("fp16", None)
    train_config.pop("fp16_opt_level", None)
    train_config.pop("remove_unused_columns", None)

    output_dir = "weights_" + run_name
    logging_dir = "logs_" + run_name
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        logging_dir=logging_dir,
        do_train=True,
        do_eval=True,
        fp16=False,
        fp16_opt_level="02",
        seed=seed,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        **train_config,
    )
    return training_args
