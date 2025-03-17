from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import AutoTokenizer
from transformers import AutoModel
from peft import LoraConfig
from peft import get_peft_model


def load_ankh_model(
    ckpt: str,
    use_lora: bool,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    bias: str,
):
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)
    if use_lora:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    return model.encoder, tokenizer


def load_esm_model(
    ckpt: str,
    use_lora: bool,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    bias: str,
):
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt)
    if use_lora:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    return model, tokenizer


def load_prott5_model(
    ckpt: str,
    use_lora: bool,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    bias: str,
):
    tokenizer = T5Tokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)
    if use_lora:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    return model.encoder, tokenizer


ankh_checkpoints = [
    "ElnaggarLab/ankh-base",
    "ElnaggarLab/ankh-large",
    "ElnaggarLab/ankh2-ext1",
    "ElnaggarLab/ankh2-ext2",
]

esm_checkpoints = [
    "facebook/esm2_t36_3B_UR50D",
    "facebook/esm2_t33_650M_UR50D",
]

prott5_checkpoints = ["Rostlab/prot_t5_xl_uniref50"]


def load_backbone(
    ckpt: str,
    use_lora: bool,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    bias: str,
):
    if ckpt in ankh_checkpoints:
        return load_ankh_model(
            ckpt=ckpt,
            use_lora=use_lora,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            bias=bias,
        )
    elif ckpt in esm_checkpoints:
        return load_esm_model(
            ckpt=ckpt,
            use_lora=use_lora,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            bias=bias,
        )
    elif ckpt in prott5_checkpoints:
        return load_prott5_model(
            ckpt=ckpt,
            use_lora=use_lora,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            bias=bias,
        )
    else:
        known_checkpoints = (
            ankh_checkpoints + esm_checkpoints + prott5_checkpoints
        )
        raise ValueError(
            f"Unknown checkpoint: {ckpt}, "
            f"valid options are: {known_checkpoints}"
        )
