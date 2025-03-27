from torch import nn
import torch
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        activation: str,
        gated: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.gated = gated
        self.proj_1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.proj_2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        if self.gated:
            self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.activation = getattr(F, activation)

    def reset_parameters(self):
        # Weight init
        mean = 0
        std = (2 / (self.hidden_dim + self.embed_dim)) ** 0.5
        nn.init.normal_(self.proj_1.weight, mean=mean, std=std)
        nn.init.normal_(self.proj_2.weight, mean=mean, std=std)
        if self.gated:
            nn.init.normal_(self.gate_proj.weight, mean=mean, std=std)

        # Bias init
        if self.bias:
            nn.init.zeros_(self.proj_1.bias)
            nn.init.zeros_(self.proj_2.bias)
            if self.gated:
                nn.init.zeros_(self.gate_proj.bias)

    def forward(self, embeddings: torch.FloatTensor):
        if self.gated:
            gate = self.activation(self.gate_proj(embeddings))
            output = self.proj_1(embeddings)
            output = self.proj_2(gate * output)
        else:
            output = self.activation(self.proj_1(embeddings))
            output = self.proj_2(output)
        return output
