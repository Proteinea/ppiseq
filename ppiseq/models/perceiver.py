from __future__ import annotations

import torch
from ppiseq.layers import losses
from ppiseq.layers import poolers
from ppiseq.layers.perceiver import Perceiver
from ppiseq.models.utils import BackbonePairEmbeddingExtraction
from torch import nn


class PerceiverModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        model_name: str | None = None,
        embedding_name: str | None = None,
        num_latents: int = 512,
        num_heads: int = 8,
        hidden_dim: int | None = None,
        bias: bool = False,
        num_perceiver_layers: int = 1,
        num_self_layers: int = 1,
        activation: str = "silu",
        gated: bool = False,
        shared_perceiver: bool = True,
        gradient_checkpointing: bool = False,
        loss_fn: str = "mse",
        loss_fn_options: dict = {},
    ):
        """Initialize the PerceiverModel.

        Args:
            backbone (nn.Module): The backbone model.
            pooler (nn.Module | str): The pooler.
            model_name (str | None, optional): The model name.
                Defaults to None.
            embedding_name (str | None, optional): The embedding name.
                Defaults to None.
            num_latents (int, optional): The number of latents.
                Defaults to 512.
            num_heads (int, optional): The number of heads.
                Defaults to 8.
            hidden_dim (int | None, optional): The hidden dimension.
                Defaults to None.
            bias (bool, optional): Whether to use the bias.
                Defaults to False.
            num_perceiver_layers (int, optional): The number of
                perceiver layers. Defaults to 1.
            num_self_layers (int, optional): The number of self layers.
                Defaults to 1.
            activation (str, optional): The activation function.
                Defaults to "silu".
            gated (bool, optional): Whether to use gated layer.
                Defaults to False.
            shared_perceiver (bool, optional): Whether to share the
                perceiver. Defaults to True.
            gradient_checkpointing (bool, optional): Whether to use
                gradient checkpointing. Defaults to False.
            loss_fn (str, optional): The loss function. Defaults to "mse".
            loss_fn_options (dict, optional): The options for the loss
                function. Defaults to {}.
        """
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.hidden_dim = (
            hidden_dim if hidden_dim is not None else self.embed_dim
        )

        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        self.loss_fn = losses.get(loss_fn, loss_fn_options)
        self.shared_perceiver = shared_perceiver
        if shared_perceiver:
            self.perceiver = Perceiver(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                hidden_dim=self.hidden_dim,
                num_perceiver_layers=num_perceiver_layers,
                num_self_layers=num_self_layers,
                activation=activation,
                bias=bias,
                gated=gated,
            )
        else:
            self.ligand_perceiver = Perceiver(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                hidden_dim=self.hidden_dim,
                num_perceiver_layers=num_perceiver_layers,
                num_self_layers=num_self_layers,
                activation=activation,
                bias=bias,
                gated=gated,
            )
            self.receptor_perceiver = Perceiver(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                hidden_dim=self.hidden_dim,
                num_perceiver_layers=num_perceiver_layers,
                num_self_layers=num_self_layers,
                activation=activation,
                bias=bias,
                gated=gated,
            )

        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        ligand_input_ids: torch.LongTensor,
        receptor_input_ids: torch.LongTensor,
        ligand_attention_mask: torch.LongTensor | None = None,
        receptor_attention_mask: torch.LongTensor | None = None,
        labels: torch.FloatTensor | None = None,
    ):
        ligand_embed, receptor_embed = self.backbone(
            ligand_input_ids,
            receptor_input_ids,
            ligand_attention_mask,
            receptor_attention_mask,
        )
        if self.shared_perceiver:
            output_1 = self.perceiver(
                embeddings=ligand_embed, mask=ligand_attention_mask
            )
            output_2 = self.perceiver(
                embeddings=receptor_embed, mask=receptor_attention_mask
            )
        else:
            output_1 = self.ligand_perceiver(
                embeddings=ligand_embed, mask=ligand_attention_mask
            )
            output_2 = self.receptor_perceiver(
                embeddings=receptor_embed, mask=receptor_attention_mask
            )
        output = output_1 + output_2
        pooled_output = self.pooler(output)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }
