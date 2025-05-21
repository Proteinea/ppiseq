from __future__ import annotations

import typing

import torch
from ppiseq.layers.attention import MultiHeadAttention
from ppiseq.layers.feedforward import FeedForward
from torch import nn


class SelfAttentionAndFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        activation: str,
        bias: bool = True,
        gated: bool = False,
    ):
        """Initialize the SelfAttentionAndFeedForward layer.

        Args:
            embed_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            hidden_dim (int): The dimension of the hidden layer.
            activation (str): The activation function.
            bias (bool, optional): Whether to use bias. Defaults to True.
            gated (bool, optional): Whether to use gated feedforward.
                Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.bias = bias

        self.self_attn_pre_norm = nn.RMSNorm(embed_dim)
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
        )
        self.self_ffn_pre_norm = nn.RMSNorm(embed_dim)
        self.self_ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            gated=gated,
            bias=bias,
        )

    def forward(
        self,
        latents: torch.FloatTensor,
    ):
        latents = self.self_attn_pre_norm(latents)
        latents = self.self_attn(latents, latents, None) + latents
        latents = self.self_ffn_pre_norm(latents)
        latents = self.self_ffn(latents) + latents
        return latents


class CrossAttentionAndFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        activation: str,
        bias: bool = True,
        gated: bool = False,
    ):
        """Initialize the CrossAttentionAndFeedForward layer.

        Args:
            embed_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            hidden_dim (int): The dimension of the hidden layer.
            activation (str): The activation function.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.bias = bias

        self.cross_attn_pre_norm = nn.RMSNorm(embed_dim)
        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
        )
        self.cross_ffn_pre_norm = nn.RMSNorm(embed_dim)
        self.cross_ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            gated=gated,
            bias=bias,
        )

    def forward(
        self,
        latents: torch.FloatTensor,
        embeddings: torch.FloatTensor,
        mask: torch.LongTensor | None = None,
    ):
        latents = self.cross_attn_pre_norm(latents)
        latents = self.cross_attn(latents, embeddings, mask) + latents
        latents = self.cross_ffn_pre_norm(latents)
        latents = self.cross_ffn(latents) + latents
        return latents


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_self_layers: int,
        activation: typing.Callable,
        bias: bool = True,
        gated: bool = False,
    ):
        """Initialize the PerceiverLayer.

        Args:
            embed_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            hidden_dim (int): The dimension of the hidden layer.
            num_self_layers (int): The number of self-attention layers.
            activation (typing.Callable): The activation function.
            bias (bool, optional): Whether to use bias. Defaults to True.
            gated (bool, optional): Whether to use gated feedforward.
                Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.bias = bias

        self.latent_module = CrossAttentionAndFeedForward(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            activation=activation,
            bias=bias,
            gated=gated,
        )

        self.self_layers = nn.ModuleList(
            [
                SelfAttentionAndFeedForward(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    bias=bias,
                    gated=gated,
                )
                for _ in range(num_self_layers)
            ]
        )

    def forward(
        self,
        latents: torch.FloatTensor,
        embeddings: torch.FloatTensor,
        mask: torch.LongTensor | None = None,
    ):
        latents = self.latent_module(
            latents=latents,
            embeddings=embeddings,
            mask=mask,
        )
        for self_layer in self.self_layers:
            latents = self_layer(latents)
        return latents


class Perceiver(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_latents: int,
        hidden_dim: int,
        num_perceiver_layers: int,
        num_self_layers: int,
        activation: str,
        bias: bool = True,
        gated: bool = False,
    ):
        """Initialize the Perceiver.

        Args:
            embed_dim (int): The dimension of the embeddings.
            num_heads (int): The number of attention heads.
            num_latents (int): The number of latents.
            hidden_dim (int): The dimension of the hidden layer.
            num_perceiver_layers (int): The number of perceiver layers.
            num_self_layers (int): The number of self-attention layers.
            activation (str): The activation function.
            bias (bool, optional): Whether to use bias. Defaults to True.
            gated (bool, optional): Whether to use gated feedforward.
                Defaults to False.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.latents = nn.Parameter(torch.ones(1, num_latents, embed_dim))

        self.perceiver_layers = nn.ModuleList(
            [
                PerceiverLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    num_self_layers=num_self_layers,
                    activation=activation,
                    bias=bias,
                    gated=gated,
                )
                for _ in range(num_perceiver_layers)
            ]
        )

    def forward(
        self,
        embeddings: torch.FloatTensor,
        mask: torch.LongTensor | None = None,
    ):
        latents = self.latents.expand(embeddings.shape[0], -1, -1)
        for perceiver_layer in self.perceiver_layers:
            latents = perceiver_layer(
                latents=latents,
                embeddings=embeddings,
                mask=mask,
            )
        return latents
