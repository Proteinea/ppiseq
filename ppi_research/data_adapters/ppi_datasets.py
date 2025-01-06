from __future__ import annotations
from typing import Callable
from ppi_research.data_adapters.dataset_adapters import PPIDataset
from datasets import load_dataset
from torch.utils.data import ConcatDataset


def load_skempi2_ppi_dataset(preprocessing_function: Callable | None = None):
    ds = load_dataset(
        "proteinea/skempi2_ppi_dataset", data_dir="mutation splits"
    )
    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=["protein 1 sequence", "protein 2 sequence"],
        label_column_name="affinity (pKd)",
        preprocessing_function=preprocessing_function,
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=["protein 1 sequence", "protein 2 sequence"],
        label_column_name="affinity (pKd)",
        preprocessing_function=preprocessing_function,
    )
    test_ds = PPIDataset(
        ds["test"],
        sequence_column_names=["protein 1 sequence", "protein 2 sequence"],
        label_column_name="affinity (pKd)",
        preprocessing_function=preprocessing_function,
    )
    return train_ds, {"validation": val_ds, "test": test_ds}


def load_inhouse_fc2ra(preprocessing_function: Callable | None = None):
    data_files = {
        "train": "FcR2a_R131_train.csv",
        "validation": "FcR2a_R131_test.csv",
    }
    ds = load_dataset("proteinea/inhouse-ppi-affinity", data_files=data_files)

    seq_col_names = ["Sequence", "FcR2a_R131"]
    label_name = "FcyRIIa.131R_Fold"
    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=seq_col_names,
        label_column_name=label_name,
        preprocessing_function=preprocessing_function,
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=seq_col_names,
        label_column_name=label_name,
        preprocessing_function=preprocessing_function,
    )
    return train_ds, {"validation": val_ds}


def load_inhouse_fc2rb(preprocessing_function: Callable | None = None):
    data_files = {
        "train": "FcyRIIb_train.csv",
        "validation": "FcyRIIb_test.csv",
    }
    ds = load_dataset("proteinea/inhouse-ppi-affinity", data_files=data_files)

    seq_col_names = ["Sequence", "FcR2b"]
    label_name = "FcyRIIb_Fold"
    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=seq_col_names,
        label_column_name=label_name,
        preprocessing_function=preprocessing_function,
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=seq_col_names,
        label_column_name=label_name,
        preprocessing_function=preprocessing_function,
    )
    return train_ds, {"validation": val_ds}


def load_inhouse_fc2ra_and_fc2rb_mixture(
    preprocessing_function: Callable | None = None,
):
    fc2ra_train, fc2ra_val = load_inhouse_fc2ra(preprocessing_function)
    fc2rb_train, fc2rb_val = load_inhouse_fc2rb(preprocessing_function)

    train_ds = ConcatDataset([fc2ra_train, fc2rb_train])
    val_ds = ConcatDataset([fc2ra_val["validation"], fc2rb_val["validation"]])
    return train_ds, {"validation": val_ds}


available_datasets = {
    "skempi2": load_skempi2_ppi_dataset,
    "fc2ra": load_inhouse_fc2ra,
    "fc2rb": load_inhouse_fc2rb,
    "fc_mixture": load_inhouse_fc2ra_and_fc2rb_mixture,
}


def load_ppi_dataset(identifier, *args, **kwargs):
    return available_datasets[identifier](*args, **kwargs)
