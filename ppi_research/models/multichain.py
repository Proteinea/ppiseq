from __future__ import annotations
from torch import nn
import torch

from ppi_research.models.utils import BackbonePairEmbeddingExtraction


def aggregate_chains(chain_1, chain_2, aggregation_method: str):
    if aggregation_method == "concat":
        return torch.cat([chain_1, chain_2], dim=-1)
    elif aggregation_method == "mean":
        return (chain_1 + chain_2) / 2
    elif aggregation_method == "add":
        return chain_1 + chain_2
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")


class MultiChainModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        sequence_pooler: nn.Module,
        chains_pooler: nn.Module,
        model_name: str,
        embedding_name: str,
        aggregation_method: str = "concat",
        use_ffn: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.use_ffn = use_ffn
        self.aggregation_method = aggregation_method
        input_dim = (
            self.embed_dim * 2
            if self.aggregation_method == "concat"
            else self.embed_dim
        )
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
        )
        self.sequence_pooler = sequence_pooler
        self.chains_pooler = chains_pooler

        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=bias),
                nn.SiLU(),
                nn.Linear(input_dim, input_dim, bias=bias),
            )

        self.output = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def process_chains(
        self,
        protein_embed: torch.FloatTensor,
        chain_ids: torch.LongTensor,
    ) -> torch.FloatTensor:
        pooled_chains = [
            self.chains_pooler(
                protein_embed[chain_ids == chain_id, ...], dim=0,
            )
            for chain_id in torch.unique(chain_ids)
        ]

        return torch.stack(pooled_chains, dim=0)

    def forward(
        self,
        input_ids_1: torch.LongTensor,
        input_ids_2: torch.LongTensor,
        attention_mask_1: torch.LongTensor | None = None,
        attention_mask_2: torch.LongTensor | None = None,
        chain_ids_1: torch.LongTensor | None = None,
        chain_ids_2: torch.LongTensor | None = None,
        labels: torch.FloatTensor | None = None,
    ):
        protein_1_embed, protein_2_embed = self.backbone(
            input_ids_1,
            input_ids_2,
            attention_mask_1,
            attention_mask_2,
        )

        pooled_sequences_1 = self.sequence_pooler(
            protein_1_embed, attention_mask_1, dim=1,
        )

        pooled_sequences_2 = self.sequence_pooler(
            protein_2_embed, attention_mask_2, dim=1,
        )

        chains_1_embed = chains_2_embed = None
        if chain_ids_1 is not None:
            chains_1_embed = self.process_chains(
                pooled_sequences_1, chain_ids_1,
            )
        else:
            chains_1_embed = pooled_sequences_1

        if chain_ids_2 is not None:
            chains_2_embed = self.process_chains(
                pooled_sequences_2, chain_ids_2
            )
        else:
            chains_2_embed = pooled_sequences_2

        chains_embed = aggregate_chains(
            chains_1_embed, chains_2_embed, self.aggregation_method
        )

        if self.use_ffn:
            chains_embed = self.ffn(chains_embed)

        logits = self.output(chains_embed)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }
