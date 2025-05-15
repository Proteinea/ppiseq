import random
from omegaconf import DictConfig
import numpy as np
import torch
from transformers import TrainingArguments
from ppi_research.models import AttnPoolAddModel
from ppi_research.models import AttnPoolAddConvBERTModel
from ppi_research.models import EmbedConcatModel
from ppi_research.models import EmbedConcatConvBERTModel
from ppi_research.models import MultiChainModel
from ppi_research.models import MultiChainConvBERTModel
from ppi_research.models import PerceiverModel
from ppi_research.models import SequenceConcatModel
from ppi_research.models import SequenceConcatConvBERTModel
from ppi_research.data_adapters.collators import PairCollator
from ppi_research.data_adapters.collators import MultiChainCollator
from ppi_research.data_adapters.collators import SequenceConcatCollator
from ppi_research.models.backbones import supported_checkpoints

arch_to_collator_map = {
    "pad": PairCollator,
    "ec": PairCollator,
    "mc": MultiChainCollator,
    "pc": PairCollator,
    "sc": SequenceConcatCollator,
}


valid_archs = list(arch_to_collator_map.keys())


def get_run_configs(cfg: DictConfig) -> str:
    """Get the run configs.

    Args:
        cfg (DictConfig): The configuration.

    Raises:
        ValueError: If the architecture is not valid.

    Returns:
        str: The run configs.
    """
    common_kwargs = dict(
        backbone=cfg.ckpt,
        setup=cfg.architecture,
        pooler=cfg.pooler,
        seed=cfg.train_config.seed,
        loss_fn=cfg.loss_config.name,
    )
    if cfg.lora_config.enable:
        common_kwargs["setup"] = (
            f"lora-{cfg.architecture}-r_{cfg.lora_config.r}-"
            f"alpha_{cfg.lora_config.alpha}-"
            f"target_modules_{cfg.lora_config.target_modules}"
        )
    elif cfg.convbert_config.enable:
        common_kwargs["setup"] = f"convbert-{cfg.architecture}"

    arch = cfg.architecture
    if arch == "pad":
        arch_kwargs = dict(
            shared_attention=cfg.attn_pool_add_config.shared_attention,
            use_ffn=cfg.attn_pool_add_config.use_ffn,
            ffn_multiplier=cfg.attn_pool_add_config.ffn_multiplier,
        )
    elif arch == "ec":
        arch_kwargs = dict(concat_first=cfg.embed_concat_config.concat_first)
    elif arch == "mc":
        arch_kwargs = dict(
            chains_pooler=cfg.multichain_config.chains_pooler,
            shared_global_pooler=cfg.multichain_config.shared_global_pooler,
            shared_chains_pooler=cfg.multichain_config.shared_chains_pooler,
            aggregation_method=cfg.multichain_config.aggregation_method,
            use_ffn=cfg.multichain_config.use_ffn,
            bias=cfg.multichain_config.bias,
        )
    elif arch == "pc":
        arch_kwargs = dict(
            num_latents=cfg.perceiver_config.num_latents,
            num_heads=cfg.perceiver_config.num_heads,
            hidden_dim=cfg.perceiver_config.hidden_dim,
            bias=cfg.perceiver_config.bias,
            num_perceiver_layers=cfg.perceiver_config.num_perceiver_layers,
            num_self_layers=cfg.perceiver_config.num_self_layers,
            activation=cfg.perceiver_config.activation,
            gated=cfg.perceiver_config.gated,
            shared_perceiver=cfg.perceiver_config.shared_perceiver,
        )
    elif arch == "sc":
        # No architecture specific kwargs for SC model
        arch_kwargs = dict()
    else:
        raise ValueError(
            f"Unknown architecture: {arch}, "
            f"Valid architectures: {valid_archs}"
        )
    return common_kwargs | arch_kwargs


def create_run_name(cfg: DictConfig) -> str:
    """Create a run name from the configuration.

    Args:
        cfg (DictConfig): The configuration.

    Returns:
        str: The run name.
    """
    kwargs = get_run_configs(cfg)
    output = ""
    for k, v in kwargs.items():
        if isinstance(v, list):
            v = "_".join(v)
        output += f"{k}_{v}-"
    return output[:-1]


def add_lora_prefix(setup: str, use_lora: bool) -> str:
    """Add a prefix to the setup string based on whether LORA is used.

    Args:
        setup (str): The setup string.
        use_lora (bool): Whether LORA is used.

    Returns:
        str: The setup string with the prefix.
    """
    if use_lora:
        return "lora_" + setup
    else:
        return "ft_" + setup


def set_seed(seed: int):
    """Set the seed for the random number generators.

    Args:
        seed (int): The seed.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def get_default_training_args(run_name: str, **train_config):
    """Get the default training arguments.

    Args:
        run_name (str): The run name.

    Returns:
        TrainingArguments: The training arguments.
    """
    train_config.pop("output_dir", None)
    train_config.pop("run_name", None)
    train_config.pop("logging_dir", None)
    train_config.pop("do_train", None)
    train_config.pop("do_eval", None)
    train_config.pop("fp16", None)
    train_config.pop("fp16_opt_level", None)
    train_config.pop("remove_unused_columns", None)
    seed = train_config.pop("seed", 7)
    train_config.pop("load_best_model_at_end", None)

    output_dir = "weights/" + run_name
    logging_dir = "logs/" + run_name
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


def get_ppi_downstream_model(
    backbone: torch.nn.Module,
    model_name: str,
    cfg: DictConfig,
    embedding_name: str,
):
    """Get the PPI downstream model.

    Args:
        backbone (torch.nn.Module): The backbone model.
        model_name (str): The model name.
        cfg (DictConfig): The configuration.
        embedding_name (str): The embedding name.

    Raises:
        ValueError: If the architecture is not valid.

    Returns:
        torch.nn.Module: The PPI downstream model.
    """
    arch = cfg.architecture
    common_kwargs = dict(
        backbone=backbone,
        pooler=cfg.pooler,
        model_name=model_name,
        embedding_name=embedding_name,
        loss_fn=cfg.loss_config.name,
        loss_fn_options=cfg.loss_config.options,
    )
    if arch == "pad":
        kwargs = (
            dict(
                shared_attention=cfg.attn_pool_add_config.shared_attention,
                use_ffn=cfg.attn_pool_add_config.use_ffn,
                ffn_multiplier=cfg.attn_pool_add_config.ffn_multiplier,
            )
            | common_kwargs
        )
        if cfg.convbert_config.enable:
            return AttnPoolAddConvBERTModel(
                shared_convbert=cfg.attn_pool_add_config.shared_convbert,
                convbert_dropout=cfg.convbert_config.convbert_dropout,
                convbert_attn_dropout=cfg.convbert_config.convbert_attn_dropout, # noqa
                **kwargs,
            )
        return AttnPoolAddModel(
            **kwargs,
            gradient_checkpointing=cfg.enable_gradient_checkpointing,
        )
    elif arch == "ec":
        kwargs = (
            dict(concat_first=cfg.embed_concat_config.concat_first)
            | common_kwargs
        )
        if cfg.convbert_config.enable:
            return EmbedConcatConvBERTModel(
                convbert_dropout=cfg.convbert_config.convbert_dropout,
                convbert_attn_dropout=cfg.convbert_config.convbert_attn_dropout, # noqa
                **kwargs,
            )
        return EmbedConcatModel(
            **kwargs,
            gradient_checkpointing=cfg.enable_gradient_checkpointing,
        )
    elif arch == "mc":
        kwargs = (
            dict(
                global_pooler=common_kwargs.pop("pooler"),
                chains_pooler=cfg.multichain_config.chains_pooler,
                shared_global_pooler=cfg.multichain_config.shared_global_pooler, # noqa
                shared_chains_pooler=cfg.multichain_config.shared_chains_pooler, # noqa
                aggregation_method=cfg.multichain_config.aggregation_method,
                use_ffn=cfg.multichain_config.use_ffn,
                bias=cfg.multichain_config.bias,
            )
            | common_kwargs
        )
        if cfg.convbert_config.enable:
            return MultiChainConvBERTModel(
                shared_convbert=cfg.attn_pool_add_config.shared_convbert,
                convbert_dropout=cfg.convbert_config.convbert_dropout,
                convbert_attn_dropout=cfg.convbert_config.convbert_attn_dropout, # noqa
                **kwargs,
            )
        return MultiChainModel(
            **kwargs,
            gradient_checkpointing=cfg.enable_gradient_checkpointing,
        )
    elif arch == "pc":
        return PerceiverModel(
            num_latents=cfg.perceiver_config.num_latents,
            num_heads=cfg.perceiver_config.num_heads,
            hidden_dim=cfg.perceiver_config.hidden_dim,
            bias=cfg.perceiver_config.bias,
            num_perceiver_layers=cfg.perceiver_config.num_perceiver_layers,
            num_self_layers=cfg.perceiver_config.num_self_layers,
            activation=cfg.perceiver_config.activation,
            gated=cfg.perceiver_config.gated,
            shared_perceiver=cfg.perceiver_config.shared_perceiver,
            **common_kwargs,
        )
    elif arch == "sc":
        if cfg.convbert_config.enable:
            return SequenceConcatConvBERTModel(
                convbert_dropout=cfg.convbert_config.convbert_dropout,
                convbert_attn_dropout=cfg.convbert_config.convbert_attn_dropout, # noqa
                **common_kwargs,
            )
        return SequenceConcatModel(
            **common_kwargs,
            gradient_checkpointing=cfg.enable_gradient_checkpointing,
        )
    else:
        raise ValueError(
            f"Unknown architecture: {arch}, "
            f"Valid architectures: {valid_archs}"
        )


def get_collator_cls(identifier: str):
    """Get the collator class.

    Args:
        identifier (str): The identifier.

    Raises:
        ValueError: If the identifier is not valid.

    Returns:
        Callable: The collator class.
    """
    if identifier not in valid_archs:
        raise ValueError(
            f"Unknown architecture: {identifier}, "
            f"Valid architectures: {valid_archs}"
        )
    return arch_to_collator_map[identifier]


def validate_config(cfg: DictConfig):
    """Validate the configuration.

    Args:
        cfg (DictConfig): The configuration.

    Raises:
        ValueError: If the architecture is not valid.
        ValueError: If the ckpt is not supported.
        ValueError: If LORA and ConvBERT are enabled at the same time.
        ValueError: If the Perceiver model does not support ConvBERT.
    """
    if cfg.architecture not in valid_archs:
        raise ValueError(
            f"Unknown architecture: {cfg.architecture}, "
            f"Valid architectures: {valid_archs}"
        )

    if cfg.ckpt not in supported_checkpoints:
        raise ValueError(
            f"Unsupported ckpt: {cfg.ckpt}, "
            f"Available backbones: {supported_checkpoints}"
        )
    if cfg.lora_config.enable and cfg.convbert_config.enable:
        raise ValueError(
            "LORA and ConvBERT cannot be enabled at the same time"
        )
    if cfg.architecture == "pc" and cfg.convbert_config.enable:
        raise ValueError("Perceiver model does not support ConvBERT")


def get_model_name_from_ckpt(ckpt: str) -> str:
    """Get the model name from the ckpt.

    Args:
        ckpt (str): The ckpt.

    Returns:
        str: The model name.
    """
    if "prot_t5" in ckpt:
        return "prott5"
    if "esm3" in ckpt:
        return "esm3"
    if "esm" in ckpt:
        return "esm"
    if "ankh" in ckpt:
        return "ankh"
    raise ValueError(f"Unsupported ckpt: {ckpt}")


def get_model_embedding_name(ckpt: str) -> str:
    """Get the model embedding name from the ckpt.

    Args:
        ckpt (str): The ckpt.

    Raises:
        ValueError: If the ckpt is not supported.

    Returns:
        str: The model embedding name.
    """
    if "prot_t5" in ckpt:
        return "last_hidden_state"
    if "esm3" in ckpt:
        return "embeddings"
    if "esm" in ckpt:
        return "last_hidden_state"
    if "ankh" in ckpt:
        return "last_hidden_state"
    raise ValueError(f"Unsupported ckpt: {ckpt}")
