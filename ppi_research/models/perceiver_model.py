from torch import nn
from ppi_research.layers.poolers import global_mean_pooling1d
from ppi_research.layers.perceiver import Perceiver


class PerceiverModel(nn.Module):
    def __init__(
        self,
        backbone,
        num_latents=512,
        num_heads=8,
        attn_dropout=0.0,
        bias=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.config.hidden_size
        self.output = nn.Linear(self.embed_dim, 1)
        self.perceiver = Perceiver(
            embed_dim=self.embed_dim,
            num_latents=num_latents,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            bias=bias,
        )
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
        protein_1_embed = self.backbone(
            input_ids=protein_1, attention_mask=attention_mask_1
        )[0]
        protein_2_embed = self.backbone(
            input_ids=protein_2, attention_mask=attention_mask_2
        )[0]
        output_1 = self.perceiver(
            inputs=protein_1_embed, attention_mask=attention_mask_1
        )
        output_2 = self.perceiver(
            inputs=protein_2_embed, attention_mask=attention_mask_2
        )
        output = output_1 + output_2
        pooled_output = global_mean_pooling1d(output)
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
