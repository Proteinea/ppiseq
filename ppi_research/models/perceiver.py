from ppi_research.layers.perceiver import Perceiver
from ppi_research.models.utils import BackbonePairEmbeddingExtraction
from torch import nn


class PerceiverModel(nn.Module):
    def __init__(
        self,
        backbone,
        pooler,
        model_name,
        embedding_name,
        num_latents=512,
        num_heads=8,
        hidden_dim=None,
        bias=False,
        num_perceiver_layers=1,
        num_self_layers=1,
        activation="silu",
        gated=False,
        shared_perceiver=True,
    ):
        super().__init__()
        self.embed_dim = backbone.config.hidden_size
        self.hidden_dim = (
            hidden_dim if hidden_dim is not None else self.embed_dim
        )

        self.backbone = BackbonePairEmbeddingExtraction(
            backbone=backbone,
            model_name=model_name,
            embedding_name=embedding_name,
            trainable=True,
        )
        self.pooler = pooler

        if shared_perceiver:
            self.perceiver = Perceiver(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                hidden_dim=self.hidden_dim,
                num_perceiver_layers=num_perceiver_layers,
                num_self_layers=num_self_layers,
                activation=activation,
                bias=bias,
                gated=gated,
            )
        else:
            self.perceiver_1 = Perceiver(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                hidden_dim=self.hidden_dim,
                num_perceiver_layers=num_perceiver_layers,
                num_self_layers=num_self_layers,
                activation=activation,
                bias=bias,
                gated=gated,
            )
            self.perceiver_2 = Perceiver(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                num_latents=num_latents,
                hidden_dim=self.hidden_dim,
                num_perceiver_layers=num_perceiver_layers,
                num_self_layers=num_self_layers,
                activation=activation,
                bias=bias,
                gated=gated,
            )

        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

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
        if self.shared_perceiver:
            output_1 = self.perceiver(
                inputs=protein_1_embed, attention_mask=attention_mask_1
            )
            output_2 = self.perceiver(
                inputs=protein_2_embed, attention_mask=attention_mask_2
            )
        else:
            output_1 = self.perceiver_1(
                inputs=protein_1_embed, attention_mask=attention_mask_1
            )
            output_2 = self.perceiver_2(
                inputs=protein_2_embed, attention_mask=attention_mask_2
            )
        output = output_1 + output_2
        pooled_output = self.pooler(output)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
