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
    return model, tokenizer


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
    return model, tokenizer
