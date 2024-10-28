from torch import nn
from ppi_research.layers.poolers import global_mean_pooling1d


class SequenceConcatModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.config.hidden_size
        self.output = nn.Linear(self.embed_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]

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
