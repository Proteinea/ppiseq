from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import root_mean_squared_error
from transformers import EvalPrediction


def compute_spearman(p: EvalPrediction) -> float:
    """Compute the Spearman correlation coefficient.

    Args:
        p (EvalPrediction): The evaluation prediction.

    Returns:
        float: The Spearman correlation coefficient.
    """
    predictions, labels = p.predictions.reshape(-1), p.label_ids.reshape(-1)
    return spearmanr(predictions, labels).correlation


def compute_error_bar_for_regression(spearman_corr, num_examples):
    """Compute the error bar for the Spearman correlation coefficient.

    Args:
        spearman_corr (float): The Spearman correlation coefficient.
        num_examples (int): The number of examples.

    Returns:
        float: The error bar.
    """
    error_bar = (
        (1 - spearman_corr**2) ** 2
        * (1 + spearman_corr**2 / 2)
        / (num_examples - 3)
    ) ** 0.5
    return error_bar


def compute_rmse(p: EvalPrediction):
    """Compute the Root Mean Squared Error.

    Args:
        p (EvalPrediction): The evaluation prediction.

    Returns:
        float: The Root Mean Squared Error.
    """
    return root_mean_squared_error(
        p.label_ids.flatten(), p.predictions.flatten()
    )


def compute_pearsonr(p: EvalPrediction):
    """Compute the Pearson correlation coefficient.

    Args:
        p (EvalPrediction): The evaluation prediction.

    Returns:
        float: The Pearson correlation coefficient.
    """
    return pearsonr(p.predictions.flatten(), p.label_ids.flatten()).statistic


def compute_ppi_metrics(p: EvalPrediction):
    """Compute the PPI metrics.

    Args:
        p (EvalPrediction): The evaluation prediction.

    Returns:
        dict: The PPI metrics.
    """
    spearman_stat = compute_spearman(p)
    num_examples = p.label_ids.shape[0]
    error_bar = compute_error_bar_for_regression(
        spearman_corr=spearman_stat, num_examples=num_examples
    )
    rmse = compute_rmse(p)
    pearson_corr = compute_pearsonr(p)
    return {
        "spearman": spearman_stat,
        "error_bar": error_bar,
        "pearsonr": pearson_corr,
        "rmse": rmse,
    }
