from __future__ import annotations

import torch
from ppiseq.layers import losses
from ppiseq.layers import poolers
from ppiseq.layers.convbert_encoder import ConvBertEncoder
from ppiseq.models.utils import BackbonePairEmbeddingExtraction
from torch import nn


class EmbedConcatConvBERTModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        concat_first: bool = False,
        convbert_dropout: float = 0.2,
        convbert_attn_dropout: float = 0.1,
        model_name: str | None = None,
        embedding_name: str | None = None,
        loss_fn: str = "mse",
        loss_fn_options: dict = {},
    ):
        """Initialize the EmbedConcatConvBERTModel.

        Args:
            backbone (nn.Module): The backbone model.
            pooler (nn.Module | str): The pooler.
            concat_first (bool, optional): Whether to concatenate
                the embeddings. Defaults to False.
            convbert_dropout (float, optional): The dropout for the convbert.
                Defaults to 0.2.
            convbert_attn_dropout (float, optional): The attention dropout
                for the convbert. Defaults to 0.1.
            model_name (str | None, optional): The name of the model.
                Defaults to None.
            embedding_name (str | None, optional): The name of the embedding.
                Defaults to None.
            loss_fn (str, optional): The loss function. Defaults to "mse".
            loss_fn_options (dict, optional): The options for the
                loss function. Defaults to {}.
        """
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.concat_first = concat_first
        self.backbone = BackbonePairEmbeddingExtraction(
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
        hidden_dim = (
            self.embed_dim if self.concat_first else self.embed_dim * 2
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output[-1].weight.data.uniform_(-initrange, initrange)
        self.output[-1].bias.data.zero_()

    def _extract_embeddings(self, input_ids, attention_mask=None):
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
        return outputs

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

        if self.concat_first:
            embed_output = torch.cat([ligand_embed, receptor_embed], dim=1)
            concat_attn_mask = torch.cat(
                [ligand_attention_mask, receptor_attention_mask], dim=1
            )
            embed_output = self.convbert_model(
                embed_output, concat_attn_mask
            )
            pooled_output = self.pooler(embed_output, concat_attn_mask)
        else:
            ligand_embed = self.convbert_model(
                ligand_embed, ligand_attention_mask
            )
            receptor_embed = self.convbert_model(
                receptor_embed, receptor_attention_mask
            )
            pooled_ligand = self.pooler(ligand_embed, ligand_attention_mask)
            pooled_receptor = self.pooler(
                receptor_embed, receptor_attention_mask
            )
            pooled_output = torch.cat(
                [pooled_ligand, pooled_receptor], dim=1
            )

        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }
