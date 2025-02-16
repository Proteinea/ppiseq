import torch
from ppi_research.layers import poolers
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
from transformers.models import convbert


class PoolConcatConvBERTModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        model_name: str,
        embedding_name: str,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=False,
        )
        self.pooler = poolers.get(pooler)
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
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.SiLU(),
            nn.Linear(self.embed_dim * 2, 1),
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
        ligand_input_ids,
        receptor_input_ids,
        ligand_attention_mask=None,
        receptor_attention_mask=None,
        labels=None,
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

        ligand_embed = self.convbert_layer(ligand_embed)[0]
        receptor_embed = self.convbert_layer(receptor_embed)[0]

        pooled_ligand = self.pooler(ligand_embed, ligand_attention_mask)
        pooled_receptor = self.pooler(receptor_embed, receptor_attention_mask)
        pooled_output = torch.cat([pooled_ligand, pooled_receptor], dim=1)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }


class EmbedConcatConvBERTModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pooler: nn.Module | str,
        model_name: str,
        embedding_name: str,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=False,
        )
        self.pooler = poolers.get(pooler)
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
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, 1),
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
        ligand_input_ids,
        receptor_input_ids,
        ligand_attention_mask=None,
        receptor_attention_mask=None,
        labels=None,
    ):
        ligand_embed, receptor_embed = self.backbone(
            ligand_input_ids,
            receptor_input_ids,
            ligand_attention_mask,
            receptor_attention_mask,
        )

        concat_attn_mask = None
        if (
            ligand_attention_mask is not None
            and receptor_attention_mask is not None
        ):
            ligand_attention_mask = ligand_attention_mask.to(
                device=ligand_embed.device,
                dtype=ligand_embed.dtype,
            )

            receptor_attention_mask = receptor_attention_mask.to(
                device=receptor_embed.device,
                dtype=receptor_embed.dtype,
            )

            concat_attn_mask = torch.cat(
                [ligand_attention_mask, receptor_attention_mask], dim=1
            )

        embed_output = torch.cat([ligand_embed, receptor_embed], dim=1)
        embed_output = self.convbert_layer(embed_output)[0]
        pooled_output = self.pooler(embed_output, concat_attn_mask)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
