import torch
from functools import partial


def compute_list_mle_loss(
    logits: torch.FloatTensor, targets: torch.LongTensor, margin: float = 0.5
):
    logits = logits.view(-1)
    targets = targets.view(-1)

    sorted_indices = targets.argsort(descending=True)
    sorted_logits = logits[sorted_indices]

    reversed_logits = torch.flip(sorted_logits, dims=[0])
    reversed_logcumsumexp = torch.logcumsumexp(reversed_logits, dim=0)
    suffix_logsumexp = torch.flip(reversed_logcumsumexp, dims=[0])

    loss = (suffix_logsumexp - sorted_logits).sum()
    return loss


def compute_margin_ranking_loss(
    logits: torch.FloatTensor, targets: torch.LongTensor, margin: float = 0.5
):
    rank_labels = (
        -torch.combinations(targets.squeeze(1)).diff(dim=1).sign().squeeze()
    )
    outputs = torch.combinations(logits.squeeze(-1))
    loss = torch.nn.functional.margin_ranking_loss(
        outputs[:, 0],
        outputs[:, 1],
        rank_labels,
        margin=margin,
    )
    return loss


def compute_mse_loss(logits: torch.FloatTensor, targets: torch.LongTensor):
    return torch.nn.functional.mse_loss(logits, targets)


available_losses = {
    "list_mle": compute_list_mle_loss,
    "margin_ranking": compute_margin_ranking_loss,
    "mse": compute_mse_loss,
}


def get(identifier: str, options: dict = {}):
    options = options or {}
    if identifier not in available_losses:
        available_loss_names = list(available_losses.keys())
        raise ValueError(
            "Expected loss function to be one of the "
            f"following: {available_loss_names}. Received: {identifier}."
        )
    return partial(available_losses[identifier], **options)
