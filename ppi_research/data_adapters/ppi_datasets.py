from __future__ import annotations
from typing import Callable
from ppi_research.data_adapters.dataset_adapters import PPIDataset
from datasets import load_dataset
from torch.utils.data import ConcatDataset


def load_skempi_ppi_dataset(preprocessing_function: Callable | None = None):
    ds = load_dataset("proteinea/skempi_ppi")
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


def load_ppi_deepdirect_dataset(
    preprocessing_function: Callable | None = None,
):
    ds = load_dataset("proteinea/ppi_deepdirect")
    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
        preprocessing_function=preprocessing_function,
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
        preprocessing_function=preprocessing_function,
    )
    test_ds = PPIDataset(
        ds["test"],
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
        preprocessing_function=preprocessing_function,
    )
    ds = load_dataset(
        "proteinea/ppi_deepdirect", data_files="ab-bind_processed.csv"
    )["train"]
    ab_bind_ds = PPIDataset(
        ds,
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
        preprocessing_function=preprocessing_function,
    )
    return train_ds, {
        "validation": val_ds,
        "test": test_ds,
        "ab_bind": ab_bind_ds,
    }


def load_peer_ppi_dataset(preprocessing_function: Callable | None = None):
    data_files = {
        "train": "train_split.csv",
        "validation": "valid_split.csv",
        "test": "test_split.csv",
    }
    ds = load_dataset("proteinea/peer_ppi_splits", data_files=data_files)

    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=["graph1", "graph2"],
        label_column_name="interaction",
        preprocessing_function=preprocessing_function,
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=["graph1", "graph2"],
        label_column_name="interaction",
        preprocessing_function=preprocessing_function,
    )
    test_ds = PPIDataset(
        ds["test"],
        sequence_column_names=["graph1", "graph2"],
        label_column_name="interaction",
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
    "skempi": load_skempi_ppi_dataset,
    "peer": load_peer_ppi_dataset,
    "deepdirect": load_ppi_deepdirect_dataset,
    "fc2ra": load_inhouse_fc2ra,
    "fc2rb": load_inhouse_fc2rb,
    "fc_mixture": load_inhouse_fc2ra_and_fc2rb_mixture,
}


def load_ppi_dataset(identifier, *args, **kwargs):
    return available_datasets[identifier](*args, **kwargs)
