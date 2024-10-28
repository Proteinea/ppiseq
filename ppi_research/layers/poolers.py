import torch


def global_mean_pooling1d(
    x: torch.FloatTensor, padding_mask: torch.FloatTensor = None
):
    if padding_mask is None:
        return torch.mean(x, dim=1)

    x_masked = x * padding_mask.unsqueeze(-1)
    return x_masked.sum(1) / padding_mask.sum(1)
