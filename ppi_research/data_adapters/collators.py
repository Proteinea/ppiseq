from __future__ import annotations
from typing import Callable
import torch
import random


class PairCollator:
    def __init__(self, tokenizer: Callable):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        seqs_1, seqs_2, labels = [], [], []
        for b in batch:
            seqs_1.append(b["protein_1"])
            seqs_2.append(b["protein_2"])
            labels.append(b["affinity"])

        seqs_1_encoded = self.tokenizer(
            seqs_1,
            add_special_tokens=True,
            max_length=None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        seqs_2_encoded = self.tokenizer(
            seqs_2,
            add_special_tokens=True,
            max_length=None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "protein_1": seqs_1_encoded["input_ids"],
            "protein_2": seqs_2_encoded["input_ids"],
            "attention_mask_1": seqs_1_encoded["attention_mask"],
            "attention_mask_2": seqs_2_encoded["attention_mask"],
            "labels": labels,
        }


class SequenceConcatCollator:
    def __init__(
        self,
        tokenizer,
        random_swapping=False,
        swapping_prob=0.5,
        preprocessing_function: Callable | None = None,
    ):
        if preprocessing_function is not None and not callable(
            self.preprocessing_function
        ):
            raise ValueError("`preprocessing_function` is not callable.")

        self.tokenizer = tokenizer
        self.random_swapping = random_swapping
        self.swapping_prob = swapping_prob
        self.preprocessing_function = preprocessing_function

    def __call__(self, batch):
        sequences, labels = [], []

        for b in batch:
            if self.random_swapping:
                if random.random() < self.swapping_prob:
                    inputs = (b["protein_1"], b["protein_2"])
                else:
                    inputs = (b["protein_2"], b["protein_1"])
            else:
                inputs = (b["protein_1"], b["protein_2"])

            if self.preprocessing_function is not None:
                inputs = self.preprocessing_function(inputs)

            sequences.append(inputs)
            labels.append(b["affinity"])

        encoded_sequences = self.tokenizer(
            sequences,
            add_special_tokens=True,
            max_length=None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

        return {
            "input_ids": encoded_sequences["input_ids"],
            "attention_mask": encoded_sequences["attention_mask"],
            "labels": labels,
        }
