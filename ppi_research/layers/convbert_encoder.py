from __future__ import annotations

import torch
from transformers.models import convbert
from torch import nn


class ConvBertEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
        activation: str = "gelu",
        head_ratio: int = 2,
        num_groups: int = 1,
    ):
        """
        ConvBertEncoder model.

        Args:
            input_dim: Dimension of the input embeddings.
            num_heads: Integer specifying the number of attention heads.
            hidden_dim: Integer specifying the hidden dimension.
            num_layers: Integer specifying the number of layers.
            kernel_size: Integer specifying the filter size. Default: 7.
            dropout: Float specifying the dropout rate. Default: 0.2.
            attn_dropout: Float specifying the attention dropout rate.
            Default: 0.2.
            activation: String specifying the activation function.
            Default: "gelu".
            head_ratio: Integer specifying the head ratio. Default: 2.
            num_groups: Integer specifying the number of groups. Default: 1.
        """
        super().__init__()

        config = convbert.ConvBertConfig(
            hidden_size=input_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim,
            hidden_act=activation,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=attn_dropout,
            head_ratio=head_ratio,
            conv_kernel_size=kernel_size,
            num_groups=num_groups,
        )
        self.convbert_encoder = convbert.ConvBertModel(config).encoder

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.FloatTensor:
        attention_mask = attention_mask.to(dtype=dtype, device=device)
        ext_attention_mask = attention_mask[:, None, None, :]
        ext_attention_mask = (
            1.0 - ext_attention_mask
        ) * torch.finfo(dtype).min
        return ext_attention_mask

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.LongTensor,
    ) -> torch.FloatTensor:
        ext_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            embeddings.dtype,
            embeddings.device,
        )
        hidden_states = self.convbert_encoder(
            embeddings, attention_mask=ext_attention_mask
        )[0]
        return hidden_states
