from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def freezed_forward(model: nn.Module, model_inputs: Dict):
    """Freeze the model and forward the model.

    Args:
        model (nn.Module): The model.
        model_inputs (Dict): The model inputs.

    Returns:
        The model outputs.
    """
    if model.training:
        model.eval()

    with torch.no_grad():
        return model(**model_inputs)


def extract_embeddings(
    model: nn.Module,
    model_inputs: Dict,
    trainable: bool,
    embedding_name: str | None = None,
):
    """Extract the embeddings from the model.

    Args:
        model (nn.Module): The model.
        model_inputs (Dict): The model inputs.
        trainable (bool): Whether the model is trainable.
        embedding_name (str | None, optional): The embedding name.
            Defaults to None.

    Raises:
        ValueError: If the model inputs are not a dictionary.

    Returns:
        The embeddings.
    """
    if not isinstance(model_inputs, dict):
        raise ValueError(
            "Expected `model_inputs` to be of type `dict`. "
            f"Received: {type(model_inputs)}."
        )

    if trainable:
        output = model(**model_inputs)
    else:
        output = freezed_forward(model, model_inputs)

    if embedding_name is not None:
        return getattr(output, embedding_name)
    return output


def preprocess_inputs(
    sequence: torch.LongTensor,
    attention_mask: torch.LongTensor | None = None,
    model_name: str | None = None,
):
    """Preprocess the inputs.

    Args:
        sequence (torch.LongTensor): The sequence.
        attention_mask (torch.LongTensor | None, optional): The attention mask.
            Defaults to None.
        model_name (str | None, optional): The model name. Defaults to None.

    Returns:
        The preprocessed inputs.
    """
    if model_name == "esm3":
        return {"sequence_tokens": sequence}
    else:
        return {"input_ids": sequence, "attention_mask": attention_mask}


def freeze_parameters(model: nn.Module):
    """Freeze the parameters of the model.

    Args:
        model (nn.Module): The model.
    """
    for p in model.parameters():
        p.requires_grad_(False)


class BackbonePairEmbeddingExtraction(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        model_name: str | None = None,
        embedding_name: str | None = None,
        trainable: bool = True,
        gradient_checkpointing: bool = False,
        normalize: bool = False,
    ):
        """Initialize the BackbonePairEmbeddingExtraction.

        Args:
            backbone (nn.Module): The backbone.
            model_name (str | None, optional): The model name.
                Defaults to None.
            embedding_name (str | None, optional): The embedding name.
                Defaults to None.
            trainable (bool, optional): Whether the model is trainable.
                Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use
                gradient checkpointing. Defaults to False.
            normalize (bool, optional): Whether to normalize the embeddings.
                Defaults to False.
        """
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.trainable = trainable
        self.gradient_checkpointing = gradient_checkpointing
        self.normalize = normalize

        if self.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if not self.trainable:
            freeze_parameters(self.backbone)

    def forward(
        self,
        ligand_input_ids: torch.LongTensor,
        receptor_input_ids: torch.LongTensor,
        ligand_attention_mask: torch.LongTensor | None = None,
        receptor_attention_mask: torch.LongTensor | None = None,
    ):
        ligand_inputs = preprocess_inputs(
            ligand_input_ids,
            ligand_attention_mask,
            model_name=self.model_name,
        )

        receptor_inputs = preprocess_inputs(
            receptor_input_ids,
            receptor_attention_mask,
            model_name=self.model_name,
        )

        ligand_embed = extract_embeddings(
            model=self.backbone,
            model_inputs=ligand_inputs,
            trainable=self.trainable,
            embedding_name=self.embedding_name,
        )

        receptor_embed = extract_embeddings(
            model=self.backbone,
            model_inputs=receptor_inputs,
            trainable=self.trainable,
            embedding_name=self.embedding_name,
        )

        if self.normalize:
            ligand_embed = torch.nn.functional.normalize(
                ligand_embed, p=2, dim=-1,
            )
            receptor_embed = torch.nn.functional.normalize(
                receptor_embed, p=2, dim=-1,
            )

        return ligand_embed, receptor_embed


class BackboneConcatEmbeddingExtraction(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        model_name: str | None = None,
        embedding_name: str | None = None,
        trainable: bool = True,
        gradient_checkpointing: bool = False,
        normalize: bool = False,
    ):
        """Initialize the BackboneConcatEmbeddingExtraction.

        Args:
            backbone (nn.Module): The backbone.
            model_name (str | None, optional): The model name.
                Defaults to None.
            embedding_name (str | None, optional): The embedding name.
                Defaults to None.
            trainable (bool, optional): Whether the model is trainable.
                Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use
                gradient checkpointing. Defaults to False.
            normalize (bool, optional): Whether to normalize the embeddings.
                Defaults to False.
        """
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.trainable = trainable
        self.gradient_checkpointing = gradient_checkpointing
        self.normalize = normalize

        if self.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if not self.trainable:
            freeze_parameters(self.backbone)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
    ):
        inputs = preprocess_inputs(
            input_ids,
            attention_mask,
            model_name=self.model_name,
        )
        embed = extract_embeddings(
            model=self.backbone,
            model_inputs=inputs,
            trainable=self.trainable,
            embedding_name=self.embedding_name,
        )

        if self.normalize:
            embed = torch.nn.functional.normalize(
                embed, p=2, dim=-1,
            )

        return embed


class NaNObserver(torch.overrides.TorchFunctionMode):
    def __init__(self):
        """Initialize the NaNObserver. Used to track
        intermediate tensors and check for NaNs.

        Example usage:
        ```python
        with NaNObserver():
            model(input_ids, attention_mask)
        ```
        """
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        for idx, arg in enumerate(args):
            if torch.isnan(arg).any():
                raise ValueError(f"NaN detected in the argument {idx}")

        for key, value in kwargs.items():
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in the argument {key}")

        result = func(*args, **kwargs)

        if torch.isnan(result).any():
            raise ValueError(
                f"NaN detected in the result at function: {func.__name__}"
            )

        return result
