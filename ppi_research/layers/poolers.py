from __future__ import annotations

import torch
from torch import nn


def global_mean_pooling1d(
    x: torch.FloatTensor,
    padding_mask: torch.FloatTensor | None = None,
) -> torch.FloatTensor:
    """
    Global Mean Pooling 1D.

    Args:
        x: The input tensor.
        padding_mask: The padding mask.
        dim: The dimension to pool over.

    Returns:
        The global mean pooled tensor.
    """
    if padding_mask is None:
        return torch.mean(x, dim=1)
    padding_mask = padding_mask.to(device=x.device, dtype=x.dtype)
    padding_mask = padding_mask.unsqueeze(-1)
    x_masked = x * padding_mask
    return x_masked.sum(dim=1) / padding_mask.sum(dim=1)


def global_max_pooling1d(
    x: torch.FloatTensor,
    padding_mask: torch.FloatTensor | None = None,
):
    """
    Global Max Pooling 1D.

    Args:
        x: The input tensor.
        padding_mask: The padding mask.
        dim: The dimension to pool over.

    Returns:
        The global max pooled tensor.
    """
    if padding_mask is None:
        return torch.amax(x, dim=1)

    padding_mask = padding_mask.to(device=x.device, dtype=torch.long)
    padding_mask = padding_mask.unsqueeze(-1)
    x = x.masked_fill(padding_mask.logical_not(), -torch.inf)
    return torch.amax(x, dim=1)


class GlobalAvgPooling1D(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Global Average Pooling 1D.
        """
        super().__init__()

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
    ):
        return global_mean_pooling1d(x=x, padding_mask=padding_mask)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Global Max Pooling 1D.
        """
        super().__init__()

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
    ):
        return global_max_pooling1d(x=x, padding_mask=padding_mask)


class AttentionPooling1D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bias: bool = False,
    ):
        """
        Attention Pooling 1D.

        Args:
            embed_dim: The dimension of the embeddings.
            bias: Whether to use a bias in the linear layer.
        """
        super().__init__()
        self.w_proj = nn.Linear(embed_dim, 1, bias=bias)

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
    ):
        outputs = self.w_proj(x).squeeze(-1)

        if padding_mask is not None:
            padding_mask = padding_mask.to(device=x.device, dtype=x.dtype)
            outputs = outputs + padding_mask.log()

        probs = nn.functional.softmax(outputs, dim=-1)
        return torch.sum(x * probs.unsqueeze(-1), dim=1)


class WeightedAveragePooling1D(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = False):
        """
        Weighted Average Pooling 1D.

        Args:
            embed_dim: The dimension of the embeddings.
            bias: Whether to use a bias in the linear layer.
        """
        super().__init__()
        self.w_proj = nn.Linear(embed_dim, 1, bias=bias)

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: torch.FloatTensor | None = None,
    ):
        outputs = self.w_proj(x).squeeze(-1)
        if padding_mask is None:
            gates = nn.functional.sigmoid(outputs)
            return torch.mean(x * gates.unsqueeze(-1), dim=1)

        padding_mask = padding_mask.to(device=x.device, dtype=x.dtype)
        outputs = outputs + padding_mask.log()
        gates = nn.functional.sigmoid(outputs)
        padding_mask = padding_mask.unsqueeze(-1)
        return (x * gates.unsqueeze(-1)).sum(dim=1) / padding_mask.sum(dim=1)


available_poolers = {
    "avg": GlobalAvgPooling1D,
    "max": GlobalMaxPooling1D,
    "attn": AttentionPooling1D,
    "weighted_avg": WeightedAveragePooling1D,
    "mean": GlobalAvgPooling1D,
    "weighted_mean": WeightedAveragePooling1D,
    "attention": AttentionPooling1D,
}


def get(identifier, *args, **kwargs):
    """
    Get a pooler from the available poolers.
    """
    if isinstance(identifier, nn.Module):
        return identifier

    pooler = available_poolers.get(identifier)
    if pooler is None:
        available_pooler_names = list(available_poolers.keys())
        raise ValueError(
            "Expected `identifier` to be one of the following: "
            f"{available_pooler_names}. Received: {identifier}."
        )
    return pooler(*args, **kwargs)
