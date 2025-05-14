from __future__ import annotations
import torch
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
from ppi_research.layers import poolers
from ppi_research.layers import losses


class EmbedConcatModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        concat_first: bool = True,
        model_name: str | None = None,
        embedding_name: str | None = None,
        gradient_checkpointing: bool = False,
        loss_fn: str = "mse",
        loss_fn_options: dict = {},
    ):
        """Initialize the EmbedConcatModel.

        Args:
            backbone (nn.Module): The backbone model.
            pooler (nn.Module | str): The pooler.
            concat_first (bool, optional): Whether to concatenate
                the embeddings. Defaults to True.
            model_name (str | None, optional): The name of the model.
                Defaults to None.
            embedding_name (str | None, optional): The name of the embedding.
                Defaults to None.
            gradient_checkpointing (bool, optional): Whether to use gradient
                checkpointing. Defaults to False.
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
            trainable=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        self.loss_fn = losses.get(loss_fn, loss_fn_options)

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
        for module in self.output.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-initrange, initrange)
                module.bias.data.zero_()

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

        if self.concat_first:
            concat_embed = torch.cat([ligand_embed, receptor_embed], dim=1)
            concat_attn_mask = torch.cat(
                [ligand_attention_mask, receptor_attention_mask], dim=1
            )
            pooled_embed = self.pooler(concat_embed, concat_attn_mask)
        else:
            pooled_ligand = self.pooler(ligand_embed, ligand_attention_mask)
            pooled_receptor = self.pooler(
                receptor_embed, receptor_attention_mask
            )
            pooled_embed = torch.cat([pooled_ligand, pooled_receptor], dim=1)

        logits = self.output(pooled_embed)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }
