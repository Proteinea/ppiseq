from torch import nn
from transformers.models import convbert
from ppi_research.models.utils import BackboneConcatEmbeddingExtraction


class SequenceConcatConvBERTModel(nn.Module):
    def __init__(self, backbone, pooler, model_name, embedding_name):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.backbone = BackboneConcatEmbeddingExtraction(
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

        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed = self.backbone(input_ids, attention_mask)
        embed = self.convbert_layer(embed)[0]

        attention_mask = attention_mask.to(
            device=embed.device,
            dtype=embed.dtype,
        )
        pooled_output = self.pooler(embed, attention_mask)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
