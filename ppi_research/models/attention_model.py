from torch import nn
from ppi_research.layers.poolers import global_mean_pooling1d


class AttnPoolAddModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.config.hidden_size
        self.output = nn.Linear(self.embed_dim, 1)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            dropout=0.0,
            bias=False,
            batch_first=True,
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

        attention_mask_1 = attention_mask_1.to(
            device=protein_1_embed.device,
            dtype=protein_1_embed.dtype,
        )

        attention_mask_2 = attention_mask_2.to(
            device=protein_2_embed.device,
            dtype=protein_2_embed.dtype,
        )

        output_1, _ = self.attn(
            query=protein_1_embed,
            key=protein_2_embed,
            value=protein_2_embed,
            key_padding_mask=attention_mask_2.log(),
            need_weights=False,
        )
        output_2, _ = self.attn(
            query=protein_2_embed,
            key=protein_1_embed,
            value=protein_1_embed,
            key_padding_mask=attention_mask_1.log(),
            need_weights=False,
        )

        output_1 = output_1 + protein_1_embed
        output_2 = output_2 + protein_2_embed
        pooled_output_1 = global_mean_pooling1d(output_1, attention_mask_1)
        pooled_output_2 = global_mean_pooling1d(output_2, attention_mask_2)
        pooled_output = pooled_output_1 + pooled_output_2
        logits = self.output(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(input=logits, target=labels)

        return {
            "logits": logits,
            "loss": loss,
        }
