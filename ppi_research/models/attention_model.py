from __future__ import annotations
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
import torch
from ppi_research.layers import poolers


class AttnPoolAddModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        shared_attention: bool = True,
        use_ffn: bool = False,
        ffn_multiplier: int = 1,
        model_name: str | None = None,
        embedding_name: str | None = None,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.use_ffn = use_ffn
        self.ffn_multiplier = ffn_multiplier
        self.shared_attention = shared_attention
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        if shared_attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=0.0,
                bias=False,
                batch_first=True,
            )
        else:
            self.ligand_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=0.0,
                bias=False,
                batch_first=True,
            )
            self.receptor_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=8,
                dropout=0.0,
                bias=False,
                batch_first=True,
            )

        if use_ffn:
            intermediate_dim = self.embed_dim * self.ffn_multiplier
            self.ffn = nn.Sequential(
                nn.Linear(self.embed_dim, intermediate_dim),
                nn.SiLU(),
                nn.Linear(intermediate_dim, self.embed_dim),
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

        ligand_attention_mask = ligand_attention_mask.to(
            device=ligand_embed.device,
            dtype=ligand_embed.dtype,
        )

        receptor_attention_mask = receptor_attention_mask.to(
            device=receptor_embed.device,
            dtype=receptor_embed.dtype,
        )

        if self.shared_attention:
            output_1, _ = self.attn(
                query=ligand_embed,
                key=receptor_embed,
                value=receptor_embed,
                key_padding_mask=receptor_attention_mask.log(),
                need_weights=False,
            )
            output_2, _ = self.attn(
                query=receptor_embed,
                key=ligand_embed,
                value=ligand_embed,
                key_padding_mask=ligand_attention_mask.log(),
                need_weights=False,
            )
        else:
            output_1, _ = self.ligand_attn(
                query=ligand_embed,
                key=receptor_embed,
                value=receptor_embed,
                key_padding_mask=receptor_attention_mask.log(),
                need_weights=False,
            )
            output_2, _ = self.receptor_attn(
                query=receptor_embed,
                key=ligand_embed,
                value=ligand_embed,
                key_padding_mask=ligand_attention_mask.log(),
                need_weights=False,
            )

        ligand_embed = output_1 + ligand_embed
        receptor_embed = output_2 + receptor_embed
        pooled_ligand = self.pooler(ligand_embed, ligand_attention_mask)
        pooled_receptor = self.pooler(receptor_embed, receptor_attention_mask)
        pooled_output = pooled_ligand + pooled_receptor
        if self.use_ffn:
            pooled_output = self.ffn(pooled_output)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
