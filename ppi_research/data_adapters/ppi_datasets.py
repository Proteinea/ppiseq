from __future__ import annotations
from ppi_research.data_adapters.dataset_adapters import PPIDataset
from datasets import load_dataset
from ppi_research.data_adapters.dataset_adapters import ColumnNames


def load_ppb_affinity_dataset():
    train, validation, test = load_dataset(
        "proteinea/ppb_affinity",
        "filtered",
        split=["train", "validation", "test"],
        trust_remote_code=True,
    )

    column_names = ColumnNames(
        ligand="Ligand Sequences",
        receptor="Receptor Sequences",
        label="KD(M)",
    )

    train_ds = PPIDataset(
        train,
        column_names,
    )
    val_ds = PPIDataset(
        validation,
        column_names,
    )
    test_ds = PPIDataset(
        test,
        column_names,
    )

    return train_ds, {"validation": val_ds, "test": test_ds}


available_datasets = {
    "ppb_affinity": load_ppb_affinity_dataset,
}


def load_ppi_dataset(identifier, *args, **kwargs):
    return available_datasets[identifier](*args, **kwargs)
