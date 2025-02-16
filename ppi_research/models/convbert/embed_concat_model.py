import torch
from ppi_research.layers import poolers
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
from transformers.models import convbert


class EmbedConcatConvBERTModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        concat_first: bool = False,
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
            trainable=False,
        )
        self.pooler = poolers.get(pooler, self.embed_dim)
        convbert_config = convbert.ConvBertConfig(
            hidden_size=self.embed_dim,
            num_hidden_layers=1,
            num_attention_heads=8,
            intermediate_size=self.embed_dim // 2,
            conv_kernel_size=7,
        )
        # We use only one convbert layer in
        # our benchmarking so we just use `ConvBertLayer`.
        self.convbert_layer = convbert.ConvBertLayer(convbert_config)
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
            embed_output = self.convbert_layer(embed_output)[0]
            pooled_output = self.pooler(embed_output, concat_attn_mask)
        else:
            ligand_embed = self.convbert_layer(ligand_embed)[0]
            receptor_embed = self.convbert_layer(receptor_embed)[0]
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
