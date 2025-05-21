from __future__ import annotations

import typing

from datasets import load_dataset
from ppiseq.data_adapters.dataset_adapters import ColumnNames
from ppiseq.data_adapters.dataset_adapters import PPIDataset


def load_ppb_affinity_dataset(name: str):
    """Load the PPB affinity dataset.

    Args:
        name (str): The name of the dataset.

    Returns:
        tuple: The train, validation, and test datasets.
    """
    train, validation, test = load_dataset(
        "proteinea/ppb_affinity",
        name,
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


def load_ppi_dataset(
    identifier: str,
    *args: typing.Any,
    **kwargs: typing.Any,
) -> typing.Tuple[PPIDataset, typing.Dict[str, PPIDataset]]:
    """Load the PPI dataset.

    Args:
        identifier (str): The identifier of the dataset.

    Returns:
        tuple: The train, validation, and test datasets.
    """
    if identifier not in available_datasets:
        raise ValueError(f"Dataset {identifier} not found.")
    return available_datasets[identifier](*args, **kwargs)
