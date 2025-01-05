import torch
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn
from transformers.models import convbert


class EmbedConcatConvBERTModel(nn.Module):
    def __init__(self, backbone, pooler, model_name, embedding_name):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=False,
        )
        self.pooler = pooler
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
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        for module in self.output.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-initrange, initrange)
                module.bias.data.zero_()

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

        pooled_output_1 = self.pooler(protein_1_embed, attention_mask_1)
        pooled_output_2 = self.pooler(protein_2_embed, attention_mask_2)
        pooled_output = torch.cat([pooled_output_1, pooled_output_2], dim=1)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
