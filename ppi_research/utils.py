import torch
import random
import numpy as np

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
