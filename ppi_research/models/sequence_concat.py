from __future__ import annotations
from ppi_research.models.utils import BackboneConcatEmbeddingExtraction
from torch import nn
import torch
from ppi_research.layers import poolers
from ppi_research.layers import losses


class SequenceConcatModel(nn.Module):
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
        """Initialize the SequenceConcatModel.

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
        self.backbone = BackboneConcatEmbeddingExtraction(
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
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.FloatTensor | None = None,
    ):
        embed = self.backbone(input_ids, attention_mask)
        pooled_output = self.pooler(embed, attention_mask)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
        }
