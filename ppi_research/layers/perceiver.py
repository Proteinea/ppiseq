import torch
from torch import nn


class Perceiver(nn.Module):
    def __init__(
        self, embed_dim, num_latents, num_heads=8, attn_dropout=0.0, bias=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.bias = bias
        self.latents = nn.Parameter(torch.ones((num_latents, embed_dim)))

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=bias,
            batch_first=True,
        )

    def forward(self, inputs, attention_mask=None):
        batch_size = inputs.shape[0]
        latents = self.latents.unsqueeze(dim=0)
        latents = torch.repeat_interleave(latents, batch_size, dim=0)

        attention_mask = attention_mask.to(
            dtype=inputs.dtype,
            device=inputs.device,
        )

        # attention_mask: [B, T]
        # latents_attn_mask: [B, T]
        latents_attn_mask = torch.ones(
            (batch_size, self.num_latents),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((latents_attn_mask, attention_mask), dim=-1)
        inputs = torch.cat((latents, inputs), dim=1)

        outputs, _ = self.attn(
            query=latents,
            key=inputs,
            value=inputs,
            key_padding_mask=attention_mask.log(),
            need_weights=False,
        )
        return outputs
