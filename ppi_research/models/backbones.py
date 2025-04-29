from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import AutoTokenizer
from transformers import AutoModel
from peft import LoraConfig
from peft import get_peft_model
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.pretrained import ESM3_sm_open_v0
from collections import namedtuple


def load_ankh_model(
    ckpt: str,
    use_lora: bool = False,
    rank: int | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    target_modules: list[str] | None = None,
    bias: str | None = None,
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
    use_lora: bool = False,
    rank: int | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    target_modules: list[str] | None = None,
    bias: str | None = None,
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
    use_lora: bool = False,
    rank: int | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    target_modules: list[str] | None = None,
    bias: str | None = None,
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


def load_esm3_model(
    ckpt: str = "esm3_sm_open_v0",
    use_lora: bool = False,
    rank: int | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    target_modules: list[str] | None = None,
    bias: str | None = None,
):
    tokenizer = EsmSequenceTokenizer()
    model = ESM3_sm_open_v0()
    if use_lora:
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=bias,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

    # Hardcoded hidden size for ESM3 because the
    # model is not a HuggingFace model
    model.config = namedtuple("Config", "hidden_size")(1536)
    return model, tokenizer


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

esm3_checkpoints = ["esm3_sm_open_v1"]

supported_checkpoints = (
    ankh_checkpoints + esm_checkpoints + prott5_checkpoints + esm3_checkpoints
)


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
    elif ckpt in esm3_checkpoints:
        return load_esm3_model(
            ckpt=ckpt,
            use_lora=use_lora,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            bias=bias,
        )
    else:
        raise ValueError(
            f"Unknown checkpoint: {ckpt}, "
            f"valid options are: {supported_checkpoints}"
        )
