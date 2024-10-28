from transformers import EvalPrediction
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import root_mean_squared_error


def compute_spearman(p: EvalPrediction) -> float:
    predictions, labels = p.predictions.reshape(-1), p.label_ids.reshape(-1)
    return spearmanr(predictions, labels).correlation


def compute_error_bar_for_regression(spearman_corr, num_examples):
    error_bar = (
        (1 - spearman_corr**2) ** 2
        * (1 + spearman_corr**2 / 2)
        / (num_examples - 3)
    ) ** 0.5
    return error_bar


def compute_rmse(p: EvalPrediction):
    return root_mean_squared_error(
        p.label_ids.flatten(), p.predictions.flatten()
    )


def compute_pearsonr(p: EvalPrediction):
    return pearsonr(p.predictions.flatten(), p.label_ids.flatten()).statistic


def compute_ppi_metrics(p: EvalPrediction):
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
