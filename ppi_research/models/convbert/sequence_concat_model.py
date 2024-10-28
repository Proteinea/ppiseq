from torch import nn
from transformers.models import convbert
import torch
from ppi_research.layers.poolers import global_mean_pooling1d


class SequenceConcatConvBERTModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
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

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed = self._extract_embeddings(input_ids, attention_mask)
        embed = self.convbert_layer(embed)[0]

        attention_mask = attention_mask.to(
            device=embed.device,
            dtype=embed.dtype,
        )
        pooled_output = global_mean_pooling1d(embed, attention_mask)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
