import argparse
import random

import numpy as np
import torch

from ppi_research.data_adapters import ppi_datasets

available_esm_checkpoints = {
    "esm_650m": "facebook/esm2_t33_650M_UR50D",
    "esm_3b": "facebook/esm2_t36_3B_UR50D",
}

available_ankh_checkpoints = {
    "ankh_base": "ElnaggarLab/ankh-base",
    "ankh_large": "ElnaggarLab/ankh-large",
    "ankh2_ext1": "ElnaggarLab/ankh2-ext1",
    "ankh2_ext2": "ElnaggarLab/ankh2-ext2",
}


available_prott5_checkpoints = {
    "prott5": "Rostlab/prot_t5_xl_uniref50",
}


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


def ankh_checkpoints():
    return list(available_ankh_checkpoints.keys())


def ankh_checkpoint_mapping(name):
    return available_ankh_checkpoints[name]


def esm_checkpoints():
    return list(available_esm_checkpoints.keys())


def esm_checkpoint_mapping(name):
    return available_esm_checkpoints[name]


def prott5_checkpoints():
    return list(available_prott5_checkpoints.keys())


def prott5_checkpoint_mapping(name):
    return available_prott5_checkpoints[name]


def parse_common_args(checkpoints):
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        choices=checkpoints,
    )
    argparser.add_argument(
        "--ds_name",
        type=str,
        required=True,
        choices=list(ppi_datasets.available_datasets.keys()),
    )
    argparser.add_argument(
        "--max_length",
        type=int,
        default=None,
        required=False,
    )
    argparser.add_argument(
        "--pooler",
        type=str,
        default="avg",
        required=False,
    )
    args = argparser.parse_args()
    return args
