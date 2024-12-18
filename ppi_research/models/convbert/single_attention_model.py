from torch import nn
from transformers.models import convbert
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
import torch


class AttnPoolAddConvBERTModel(nn.Module):
    def __init__(self, backbone, pooler, model_name, embedding_name):
        super().__init__()
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=False,
        )
        self.pooler = pooler
        self.embed_dim = self.backbone.config.hidden_size

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
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def _extract_embeddings(self, input_ids, attention_mask=None):
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
        return outputs

    def forward(
        self,
        protein_1,
        protein_2,
        attention_mask_1=None,
        attention_mask_2=None,
        labels=None,
    ):
        protein_1_embed, protein_2_embed = self.backbone(
            protein_1,
            protein_2,
            attention_mask_1,
            attention_mask_2,
        )

        attention_mask_1 = attention_mask_1.to(
            device=protein_1_embed.device,
            dtype=protein_1_embed.dtype,
        )

        attention_mask_2 = attention_mask_2.to(
            device=protein_2_embed.device,
            dtype=protein_2_embed.dtype,
        )

        protein_1_embed = self.convbert_layer(protein_1_embed)[0]
        protein_2_embed = self.convbert_layer(protein_2_embed)[0]

        output, _ = self.attn(
            query=protein_1_embed,
            key=protein_2_embed,
            value=protein_2_embed,
            key_padding_mask=attention_mask_2.log(),
            need_weights=False,
        )

        output = output + protein_1_embed
        pooled_output = self.pooler(output, attention_mask_1)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
