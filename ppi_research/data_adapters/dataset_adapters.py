from __future__ import annotations
import typing
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset


class PPIDataset(TorchDataset):
    def __init__(
        self,
        hf_ds: HFDataset,
        sequence_column_names: typing.List[str],
        label_column_name: str,
        preprocessing_function: typing.Callable | None = None,
    ):
        num_cols = len(sequence_column_names)
        if num_cols != 2:
            raise ValueError(
                f"Expected a list of two strings where each string represents "
                f"a column name. Received: {num_cols}"
            )

        if preprocessing_function is not None and not callable(
            self.preprocessing_function
        ):
            raise ValueError("`preprocessing_function` is not callable.")

        self.hf_ds = hf_ds
        self.sequence_column_names = sequence_column_names
        self.label_column_name = label_column_name
        self.preprocessing_function = preprocessing_function

    def __len__(self):
        return self.hf_ds.num_rows

    def __getitem__(self, idx):
        seq_1 = self.hf_ds[self.sequence_column_names[0]][idx]
        seq_2 = self.hf_ds[self.sequence_column_names[1]][idx]
        label = self.hf_ds[self.label_column_name][idx]

        if self.preprocessing_function is not None:
            seq_1 = self.preprocessing_function(seq_1)
            seq_2 = self.preprocessing_function(seq_2)

        return {
            "affinity": label,
            "protein_1": seq_1,
            "protein_2": seq_2,
        }
