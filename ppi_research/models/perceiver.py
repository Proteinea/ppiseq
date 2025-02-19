from __future__ import annotations
from ppi_research.layers.perceiver import Perceiver
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
import torch
from ppi_research.layers import poolers


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
    ):
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
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
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
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
