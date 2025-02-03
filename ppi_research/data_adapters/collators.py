from __future__ import annotations
from typing import Callable, Dict, Any
import torch
import random


class PairCollator:
    def __init__(
        self,
        tokenizer: Callable,
        max_length: int | None = None,
        is_split_into_words: bool = False,
        random_swapping=False,
        swapping_prob=0.5,
        preprocessing_function: Callable | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_split_into_words = is_split_into_words
        self.random_swapping = random_swapping
        self.swapping_prob = swapping_prob
        self.preprocessing_function = preprocessing_function

    def __call__(self, batch):
        seqs_1, seqs_2, labels = [], [], []
        for b in batch:
            if self.preprocessing_function is not None:
                protein_1 = self.preprocessing_function(b["protein_1"])
                protein_2 = self.preprocessing_function(b["protein_2"])
            else:
                protein_1 = b["protein_1"]
                protein_2 = b["protein_2"]

            if self.random_swapping:
                if random.random() < self.swapping_prob:
                    seqs_1.append(protein_1)
                    seqs_2.append(protein_2)
                else:
                    seqs_1.append(protein_2)
                    seqs_2.append(protein_1)
            else:
                seqs_1.append(protein_1)
                seqs_2.append(protein_2)
            labels.append(b["affinity"])

        seqs_1_encoded = self.tokenizer(
            seqs_1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=self.is_split_into_words,
        )

        seqs_2_encoded = self.tokenizer(
            seqs_2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=self.is_split_into_words,
        )

        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "protein_1": seqs_1_encoded["input_ids"],
            "protein_2": seqs_2_encoded["input_ids"],
            "attention_mask_1": seqs_1_encoded["attention_mask"],
            "attention_mask_2": seqs_2_encoded["attention_mask"],
            "labels": labels,
        }


def randomly_swap_sequences(seq_1, seq_2, swapping_prob=0.5):
    if random.random() < swapping_prob:
        return seq_2, seq_1
    else:
        return seq_1, seq_2


class SequenceConcatCollator:
    def __init__(
        self,
        tokenizer,
        random_swapping=False,
        swapping_prob=0.5,
        preprocessing_function: Callable | None = None,
        max_length: int | None = None,
        is_split_into_words: bool = False,
    ):
        if preprocessing_function is not None and not callable(
            preprocessing_function
        ):
            raise ValueError("`preprocessing_function` is not callable.")

        self.tokenizer = tokenizer
        self.random_swapping = random_swapping
        self.swapping_prob = swapping_prob
        self.preprocessing_function = preprocessing_function
        self.max_length = max_length
        self.is_split_into_words = is_split_into_words

    def __call__(self, batch):
        sequences, labels = [], []

        for b in batch:
            if self.random_swapping:
                inputs = randomly_swap_sequences(b["protein_1"], b["protein_2"], self.swapping_prob)
            else:
                inputs = (b["protein_1"], b["protein_2"])

            if self.preprocessing_function is not None:
                inputs = self.preprocessing_function(inputs)

            sequences.append(inputs)
            labels.append(b["affinity"])

        encoded_sequences = self.tokenizer(
            sequences,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=self.is_split_into_words,
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
        max_length: int | None = None,
        is_split_into_words: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_split_into_words = is_split_into_words

    def __call__(self, batch: list[Dict[str, Any]]):
        sequences_1 = []
        sequences_2 = []
        chain_ids_1 = []
        chain_ids_2 = []
        labels = []

        for i, b in enumerate(batch):
            # list of seqs
            protein_1_chains = b["protein_1_chains"]
            protein_2_chains = b["protein_2_chains"]
            num_chains_1 = len(protein_1_chains)
            num_chains_2 = len(protein_2_chains)

            if self.preprocessing_function is not None:
                protein_1_chains = [
                    self.preprocessing_function(chain)
                    for chain in protein_1_chains
                ]

                protein_2_chains = [
                    self.preprocessing_function(chain)
                    for chain in protein_2_chains
                ]

            sequences_1.extend(protein_1_chains)
            sequences_2.extend(protein_2_chains)

            chain_ids_1 += [i] * num_chains_1
            chain_ids_2 += [i] * num_chains_2
            labels.append(b["affinity"])

        sequences_1_encoded = self.tokenizer(
            sequences_1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=self.is_split_into_words,
        )

        sequences_2_encoded = self.tokenizer(
            sequences_2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="longest",
            truncation=self.max_length is not None,
            return_tensors="pt",
            is_split_into_words=self.is_split_into_words,
        )

        chain_ids_1 = torch.tensor(chain_ids_1, dtype=torch.long)
        chain_ids_2 = torch.tensor(chain_ids_2, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "input_ids_1": sequences_1_encoded["input_ids"],
            "input_ids_2": sequences_2_encoded["input_ids"],
            "attention_mask_1": sequences_1_encoded["attention_mask"],
            "attention_mask_2": sequences_2_encoded["attention_mask"],
            "chain_ids_1": chain_ids_1,
            "chain_ids_2": chain_ids_2,
            "labels": labels,
        }
