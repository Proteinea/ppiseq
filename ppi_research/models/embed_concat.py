import torch
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
from ppi_research.layers import poolers


class EmbedConcatModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        concat_first: bool = True,
        model_name: str | None = None,
        embedding_name: str | None = None,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.concat_first = concat_first
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        hidden_dim = (
            self.embed_dim * 2 if self.concat_first else self.embed_dim
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
            concat_embed = self.pooler(concat_embed, concat_attn_mask)
        else:
            pooled_ligand = self.pooler(ligand_embed, ligand_attention_mask)
            pooled_receptor = self.pooler(
                receptor_embed, receptor_attention_mask
            )
            pooled_output = torch.cat([pooled_ligand, pooled_receptor], dim=1)

        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
