from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def freezed_forward(model: nn.Module, model_inputs: Dict):
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
    return {"input_ids": sequence, "attention_mask": attention_mask}


def freeze_parameters(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


class BackbonePairEmbeddingExtraction(nn.Module):
    def __init__(
        self,
        backbone,
        model_name: str | None = None,
        embedding_name: str | None = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.trainable = trainable

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

        return ligand_embed, receptor_embed


class BackboneConcatEmbeddingExtraction(nn.Module):
    def __init__(
        self,
        backbone,
        model_name: str | None = None,
        embedding_name: str | None = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.trainable = trainable

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
        return embed
