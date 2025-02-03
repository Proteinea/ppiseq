from __future__ import annotations
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange
import typing
from torch.nn import functional as F


def softmax(x: torch.FloatTensor, dim: int):
    # from:
    # https://github.com/google/flaxformer/blame/ee62754ebe5a5eeb111493622de5537133822e3e/flaxformer/components/attention/dense_attention.py#L50 # noqa: E501
    with torch.no_grad():
        m = torch.maximum(x.amax(dim=dim, keepdim=True), torch.tensor(0.0))
    unnormalized = torch.exp(x - m)
    # equivalent to adding 1 to the softmax
    denom = unnormalized.sum(dim=dim, keepdim=True) + torch.exp(-m)
    return unnormalized / denom


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        add_one_to_softmax: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rearrange_axes = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.scale = embed_dim ** -0.5
        self.add_one_to_softmax = add_one_to_softmax

    def scaled_dot_product_attention(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.LongTensor | None = None,
    ):
        q = q * self.scale
        attn_logits = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            assert mask.ndim == 2, "mask must be 2D"
            mask = mask.to(device=attn_logits.device, dtype=attn_logits.dtype)
            mask = mask.view(mask.shape[0], 1, 1, mask.shape[-1])
            mask = mask.masked_fill(mask.logical_not(), float("-inf"))
            attn_logits = attn_logits + mask

        if self.add_one_to_softmax:
            attn_scores = softmax(attn_logits, dim=-1)
        else:
            attn_scores = F.softmax(attn_logits, dim=-1)

        attn = torch.matmul(attn_scores, v)
        return attn

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        mask: torch.LongTensor | None = None,
    ):
        xq = self.q_proj(q)
        xk, xv = self.kv_proj(kv).chunk(2, dim=-1)

        xq = self.rearrange_axes(xq)
        xk = self.rearrange_axes(xk)
        xv = self.rearrange_axes(xv)

        attn = self.scaled_dot_product_attention(xq, xk, xv, mask)

        output = rearrange(attn, "b h n d -> b n (h d)")
        output = self.o_proj(output)
        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        activation: str,
        gated: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.gated = gated
        self.proj_1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.proj_2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        if self.gated:
            self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.activation = getattr(F, activation)

    def reset_parameters(self):
        # Weight init
        mean = 0
        std = (2 / (self.hidden_dim + self.embed_dim)) ** 0.5
        nn.init.normal_(self.proj_1.weight, mean=mean, std=std)
        nn.init.normal_(self.proj_2.weight, mean=mean, std=std)
        if self.gated:
            nn.init.normal_(self.gate_proj.weight, mean=mean, std=std)

        # Bias init
        if self.bias:
            nn.init.zeros_(self.proj_1.bias)
            nn.init.zeros_(self.proj_2.bias)
            if self.gated:
                nn.init.zeros_(self.gate_proj.bias)

    def forward(self, embeddings: torch.FloatTensor):
        if self.gated:
            gate = self.activation(self.gate_proj(embeddings))
            output = self.proj_1(embeddings)
            output = self.proj_2(gate * output)
        else:
            output = self.activation(self.proj_1(embeddings))
            output = self.proj_2(output)
        return output


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
        mask: torch.LongTensor | None = None,
    ):
        latents = self.self_attn_pre_norm(latents)
        latents = self.self_attn(latents, latents, mask) + latents
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
        latents = self.latent_module(latents, embeddings, mask)
        for self_layer in self.self_layers:
            latents = self_layer(latents, mask)
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
