from __future__ import annotations

import torch
from torch import nn
from transformers.models import convbert


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
        """Initialize the ConvBertEncoder.

        Args:
            input_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            hidden_dim (int): The hidden dimension.
            num_layers (int, optional): The number of layers.
                Defaults to 1.
            kernel_size (int, optional): The filter size. Defaults to 7.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
            attn_dropout (float, optional): The attention dropout rate.
                Defaults to 0.1.
            activation (str, optional): The activation function.
                Defaults to "gelu".
            head_ratio (int, optional): The head ratio. Defaults to 2.
            num_groups (int, optional): The number of groups. Defaults to 1.
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
        """Get the extended attention mask.

        Args:
            attention_mask (torch.Tensor): The attention mask.
            dtype (torch.dtype): The dtype.
            device (torch.device): The device.

        Returns:
            torch.FloatTensor: The extended attention mask.
        """
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
