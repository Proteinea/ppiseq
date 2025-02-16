from __future__ import annotations
from ppi_research.models.utils import BackboneConcatEmbeddingExtraction
from torch import nn
import torch
from ppi_research.layers import poolers


class SequenceConcatModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        model_name: str | None = None,
        embedding_name: str | None = None,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackboneConcatEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.FloatTensor | None = None,
    ):
        embed = self.backbone(input_ids, attention_mask)
        pooled_output = self.pooler(embed, attention_mask)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
