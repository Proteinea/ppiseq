from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def freezed_forward(model: nn.Module, model_inputs: Dict):
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
    def __init__(self, backbone, model_name, embedding_name, trainable=True):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.trainable = trainable

        if not self.trainable:
            freeze_parameters(self.backbone)

    def forward(
        self,
        protein_1,
        protein_2,
        attention_mask_1=None,
        attention_mask_2=None,
    ):
        inputs_1 = preprocess_inputs(
            protein_1,
            attention_mask_1,
            model_name=self.model_name,
        )

        inputs_2 = preprocess_inputs(
            protein_2,
            attention_mask_2,
            model_name=self.model_name,
        )

        protein_1_embed = extract_embeddings(
            model=self.backbone,
            model_inputs=inputs_1,
            trainable=self.trainable,
            embedding_name=self.embedding_name,
        )

        protein_2_embed = extract_embeddings(
            model=self.backbone,
            model_inputs=inputs_2,
            trainable=self.trainable,
            embedding_name=self.embedding_name,
        )

        return protein_1_embed, protein_2_embed


class BackboneConcatEmbeddingExtraction(nn.Module):
    def __init__(self, backbone, model_name, embedding_name, trainable=True):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.trainable = trainable

        if not self.trainable:
            freeze_parameters(self.backbone)

    def forward(self, input_ids, attention_mask=None):
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
