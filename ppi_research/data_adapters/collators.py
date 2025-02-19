from __future__ import annotations
import typing
import torch
from ppi_research.data_adapters import preprocessing_pipelines


def validate_labels_preprocessing_function(
    labels_preprocessing_function,
) -> None:
    if labels_preprocessing_function is not None and not callable(
        labels_preprocessing_function
    ):
        raise ValueError("`labels_preprocessing_function` is not callable.")


class PairCollator:
    def __init__(
        self,
        tokenizer: typing.Callable,
        model_name: str,
        max_length: int | None = None,
        labels_preprocessing_function: typing.Callable | None = None,
    ):
        validate_labels_preprocessing_function(labels_preprocessing_function)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.preprocessor = (
            preprocessing_pipelines.SequencePairPreprocessingPipeline(
                model_name
            )
        )
        self.labels_preprocessing_function = labels_preprocessing_function

    def __call__(self, batch):
        ligand_sequences, receptor_sequences, labels = [], [], []
        for b in batch:
            ligand_sequence = b["ligand"]
            receptor_sequence = b["receptor"]
            label = b["affinity"]

            ligand_sequence, receptor_sequence = self.preprocessor.preprocess(
                ligand_sequence,
                receptor_sequence,
            )

            if self.labels_preprocessing_function is not None:
                label = self.labels_preprocessing_function(label)

            ligand_sequences.append(ligand_sequence)
            receptor_sequences.append(receptor_sequence)
            labels.append(label)

        # add_special_tokens=False because special tokens are already
        # added in the preprocessing function.
        ligand_sequences_encoded = self.tokenizer(
            ligand_sequences,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=True,
        )

        receptor_sequences_encoded = self.tokenizer(
            receptor_sequences,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=True,
        )

        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "ligand_input_ids": ligand_sequences_encoded["input_ids"],
            "receptor_input_ids": receptor_sequences_encoded["input_ids"],
            "ligand_attention_mask": ligand_sequences_encoded[
                "attention_mask"
            ],
            "receptor_attention_mask": receptor_sequences_encoded[
                "attention_mask"
            ],
            "labels": labels,
        }


class SequenceConcatCollator:
    def __init__(
        self,
        tokenizer: typing.Callable,
        model_name: str,
        max_length: int | None = None,
        labels_preprocessing_function: typing.Callable | None = None,
    ):
        validate_labels_preprocessing_function(labels_preprocessing_function)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.labels_preprocessing_function = labels_preprocessing_function
        self.preprocessor = (
            preprocessing_pipelines.SequenceConcatPreprocessingPipeline(
                model_name
            )
        )

    def __call__(self, batch):
        sequences, labels = [], []

        for b in batch:
            ligand_sequence = b["ligand"]
            receptor_sequence = b["receptor"]
            label = b["affinity"]

            sequence = self.preprocessor.preprocess(
                ligand_sequence,
                receptor_sequence,
            )

            if self.labels_preprocessing_function is not None:
                label = self.labels_preprocessing_function(label)

            sequences.append(sequence)
            labels.append(label)

        # add_special_tokens=False because special tokens are already
        # added in the preprocessing function.
        encoded_sequences = self.tokenizer(
            sequences,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=True,
        )
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "input_ids": encoded_sequences["input_ids"],
            "attention_mask": encoded_sequences["attention_mask"],
            "labels": labels,
        }


class MultiChainCollator:
    def __init__(
        self,
        tokenizer,
        model_name: str,
        max_length: int | None = None,
        labels_preprocessing_function: typing.Callable | None = None,
    ):
        validate_labels_preprocessing_function(labels_preprocessing_function)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.labels_preprocessing_function = labels_preprocessing_function
        self.preprocessor = (
            preprocessing_pipelines.MultiChainPreprocessingPipeline(model_name)
        )

    def __call__(self, batch: list[typing.Dict[str, typing.Any]]):
        ligand_sequences = []
        receptor_sequences = []
        ligand_chain_ids = []
        receptor_chain_ids = []
        labels = []

        for i, b in enumerate(batch):
            # list of seqs
            ligand_sequence = b["ligand"]
            receptor_sequence = b["receptor"]
            label = b["affinity"]

            (
                ligand_sequence,
                num_ligand_chains,
                receptor_sequence,
                num_receptor_chains,
            ) = self.preprocessor.preprocess(
                ligand_sequence,
                receptor_sequence,
            )

            if self.labels_preprocessing_function is not None:
                label = self.labels_preprocessing_function(label)

            ligand_sequences.extend(ligand_sequence)
            receptor_sequences.extend(receptor_sequence)

            ligand_chain_ids += [i] * num_ligand_chains
            receptor_chain_ids += [i] * num_receptor_chains
            labels.append(label)

        ligand_sequences_encoded = self.tokenizer(
            ligand_sequences,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=True,
        )

        receptor_sequences_encoded = self.tokenizer(
            receptor_sequences,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=True,
        )

        ligand_chain_ids = torch.tensor(ligand_chain_ids, dtype=torch.long)
        receptor_chain_ids = torch.tensor(receptor_chain_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "ligand_input_ids": ligand_sequences_encoded["input_ids"],
            "receptor_input_ids": receptor_sequences_encoded["input_ids"],
            "ligand_attention_mask": ligand_sequences_encoded[
                "attention_mask"
            ],
            "receptor_attention_mask": receptor_sequences_encoded[
                "attention_mask"
            ],
            "ligand_chain_ids": ligand_chain_ids,
            "receptor_chain_ids": receptor_chain_ids,
            "labels": labels,
        }
