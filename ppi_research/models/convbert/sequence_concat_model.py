from __future__ import annotations
from ppi_research.layers import poolers
from ppi_research.models.utils import BackboneConcatEmbeddingExtraction
from torch import nn
from ppi_research.layers.convbert_encoder import ConvBertEncoder
import torch
from ppi_research.layers import losses


class SequenceConcatConvBERTModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        convbert_dropout: float = 0.2,
        convbert_attn_dropout: float = 0.1,
        model_name: str | None = None,
        embedding_name: str | None = None,
        loss_fn: str = "mse",
        loss_fn_options: dict = {},
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackboneConcatEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=False,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        self.loss_fn = losses.get(loss_fn, loss_fn_options)

        self.convbert_model = ConvBertEncoder(
            input_dim=self.embed_dim,
            num_heads=8,
            hidden_dim=self.embed_dim // 2,
            kernel_size=7,
            dropout=convbert_dropout,
            attn_dropout=convbert_attn_dropout,
        )

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
        embed = self.convbert_model(embed, attention_mask)

        attention_mask = attention_mask.to(
            device=embed.device,
            dtype=embed.dtype,
        )
        pooled_output = self.pooler(embed, attention_mask)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }
