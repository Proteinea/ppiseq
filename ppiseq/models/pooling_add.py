from __future__ import annotations

import torch
from ppiseq.layers import losses
from ppiseq.layers import poolers
from ppiseq.models.utils import BackbonePairEmbeddingExtraction
from torch import nn


class PoolingAdditionModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        model_name: str | None = None,
        embedding_name: str | None = None,
        gradient_checkpointing: bool = False,
        loss_fn: str = "mse",
        loss_fn_options: dict = {},
    ):
        """Initialize the PoolingAdditionModel.

        Args:
            backbone (nn.Module): The backbone model.
            pooler (nn.Module | str): The pooler.
            model_name (str | None, optional): The model name.
                Defaults to None.
            embedding_name (str | None, optional): The embedding name.
                Defaults to None.
            gradient_checkpointing (bool, optional): Whether to use
                gradient checkpointing. Defaults to False.
            loss_fn (str, optional): The loss function. Defaults to "mse".
            loss_fn_options (dict, optional): The options for the
                loss function. Defaults to {}.
        """
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        self.loss_fn = losses.get(loss_fn, loss_fn_options)
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

        pooled_ligand = self.pooler(ligand_embed, ligand_attention_mask)
        pooled_receptor = self.pooler(receptor_embed, receptor_attention_mask)
        pooled_output = pooled_ligand + pooled_receptor
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }
