from __future__ import annotations

import torch
from torch import nn


def global_mean_pooling1d(
    x: torch.FloatTensor,
    padding_mask: torch.FloatTensor | None = None,
    dim: int = 1,
):
    if padding_mask is None:
        return torch.mean(x, dim=dim)

    x_masked = x * padding_mask.unsqueeze(-1)
    return x_masked.sum(dim=dim) / padding_mask.sum(dim=dim)


def global_max_pooling1d(
    x: torch.FloatTensor,
    padding_mask: torch.FloatTensor | None = None,
    dim: int = 1,
):
    if padding_mask is None:
        return torch.amax(x, dim=dim)

    padding_mask = padding_mask.to(device=x.device, dtype=torch.long)
    x = x.masked_fill(padding_mask.logical_not(), -torch.inf)
    return torch.amax(x, dim=dim)


class GlobalAvgPooling1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
        dim: int = 1,
    ):
        return global_mean_pooling1d(x=x, padding_mask=padding_mask, dim=dim)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
        dim: int = 1,
    ):
        return global_max_pooling1d(x=x, padding_mask=padding_mask, dim=dim)


class AttentionPooling1D(nn.Module):
    def __init__(self, embed_dim: int, bias=False):
        super().__init__()
        self.w_proj = nn.Linear(embed_dim, 1, bias=bias)

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
        dim: int = 1,
    ):
        outputs = self.w_proj(x).squeeze(-1)

        if padding_mask is not None:
            padding_mask = padding_mask.to(device=x.device, dtype=torch.long)
            outputs = outputs.masked_fill(
                padding_mask.logical_not(), -torch.inf
            )

        probs = nn.functional.softmax(outputs, dim=-1)
        return torch.sum(x * probs.unsqueeze(-1), dim=dim)


class GatedPooling1D(nn.Module):
    def __init__(self, embed_dim: int, bias=False):
        super().__init__()
        self.w_proj = nn.Linear(embed_dim, 1, bias=bias)

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
        dim: int = 1,
    ):
        outputs = self.w_proj(x).squeeze(-1)

        if padding_mask is not None:
            padding_mask = padding_mask.to(device=x.device, dtype=torch.long)
            outputs = outputs.masked_fill(
                padding_mask.logical_not(), -torch.inf
            )

        gates = nn.functional.sigmoid(outputs)
        return torch.sum(x * gates.unsqueeze(-1), dim=dim)


available_poolers = {
    "avg": GlobalAvgPooling1D,
    "attn": AttentionPooling1D,
    "gated": GatedPooling1D,
}


def get(identifier, *args, **kwargs):
    pooler = available_poolers.get(identifier)
    if pooler is None:
        available_pooler_names = list(available_poolers.keys())
        raise ValueError(
            "Expected `identifier` to be one of the following: "
            f"{available_pooler_names}. Received: {identifier}."
        )
    return pooler(*args, **kwargs)
