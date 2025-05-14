from __future__ import annotations
import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F


def softmax(x: torch.FloatTensor, dim: int):
    """Softmax function with added 1 to the denominator.

    Args:
        x (torch.FloatTensor): The input tensor.
        dim (int): The dimension to apply the softmax to.

    Returns:
        torch.FloatTensor: The softmaxed tensor.
    """
    # from:
    # https://github.com/google/flaxformer/blame/ee62754ebe5a5eeb111493622de5537133822e3e/flaxformer/components/attention/dense_attention.py#L50 # noqa: E501
    with torch.no_grad():
        m = torch.maximum(x.amax(dim=dim, keepdim=True), torch.tensor(0.0))
    unnormalized = torch.exp(x - m)
    # equivalent to adding 1 to the softmax
    denom = unnormalized.sum(dim=dim, keepdim=True) + torch.exp(-m)
    return unnormalized / denom


def prepare_mask(
    mask: torch.LongTensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.FloatTensor:
    """Prepare the mask for the attention.

    Args:
        mask (torch.LongTensor): The mask.
        device (torch.device): The device.
        dtype (torch.dtype): The dtype.

    Returns:
        torch.FloatTensor: The prepared mask.
    """
    assert mask.ndim == 2, "mask must be 2D"
    bsz, seqlen = mask.shape
    mask = mask.to(device=device, dtype=dtype)
    mask = mask.view(bsz, 1, 1, seqlen)
    return mask.log()


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        add_one_to_softmax: bool = False,
        attn_dropout: float = 0.0,
        bias: bool = True,
    ):
        """Initialize the MultiHeadAttention.

        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of heads.
            add_one_to_softmax (bool, optional): Whether to add 1 to the
            denominator of the softmax. Defaults to False.
            attn_dropout (float, optional): The dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to use bias. Defaults to True.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rearrange_axes = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.scale = embed_dim ** -0.5
        self.add_one_to_softmax = add_one_to_softmax
        self.attn_dropout = attn_dropout

    def scaled_dot_product_attention(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.LongTensor | None = None,
    ):
        """Scaled dot product attention.

        Args:
            q (torch.FloatTensor): The query tensor.
            k (torch.FloatTensor): The key tensor.
            v (torch.FloatTensor): The value tensor.
            mask (torch.LongTensor | None, optional): The mask.
                Defaults to None.

        Returns:
            torch.FloatTensor: The attention output.
        """
        q = q * self.scale
        attn_logits = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            mask = prepare_mask(mask, attn_logits.device, attn_logits.dtype)
            attn_logits = attn_logits + mask

        attn_logits = nn.functional.dropout(
            attn_logits,
            p=self.attn_dropout,
            training=self.training,
        )

        if self.add_one_to_softmax:
            attn_scores = softmax(attn_logits, dim=-1)
        else:
            attn_scores = F.softmax(attn_logits, dim=-1)

        attn = torch.matmul(attn_scores, v)
        return attn

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        mask: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        """Forward pass.

        Args:
            q (torch.FloatTensor): The query tensor.
            kv (torch.FloatTensor): The key-value tensor.
            mask (torch.LongTensor | None, optional): The mask.
                Defaults to None.

        Returns:
            torch.FloatTensor: The output tensor.
        """
        xq = self.q_proj(q)
        xk, xv = self.kv_proj(kv).chunk(2, dim=-1)

        xq = self.rearrange_axes(xq)
        xk = self.rearrange_axes(xk)
        xv = self.rearrange_axes(xv)

        attn = self.scaled_dot_product_attention(xq, xk, xv, mask)

        output = rearrange(attn, "b h n d -> b n (h d)")
        output = self.o_proj(output)
        return output
