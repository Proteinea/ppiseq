from __future__ import annotations
import typing
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset


class ColumnNames(typing.NamedTuple):
    ligand: str
    receptor: str
    label: str


class PPIDataset(TorchDataset):
    def __init__(
        self,
        hf_ds: HFDataset,
        column_names: ColumnNames,
    ):
        if not isinstance(column_names, ColumnNames):
            raise ValueError(
                "column_names must be a ColumnNames object. "
                f"Received: {type(column_names)}."
            )
        self.hf_ds = hf_ds
        self.column_names = column_names

    def __len__(self):
        return self.hf_ds.num_rows

    def __getitem__(self, idx):
        ligand = self.hf_ds[self.column_names.ligand][idx]
        receptor = self.hf_ds[self.column_names.receptor][idx]
        label = self.hf_ds[self.column_names.label][idx]

        if not isinstance(label, float):
            label = float(label)

        return {
            "affinity": label,
            "ligand": ligand,
            "receptor": receptor,
        }
