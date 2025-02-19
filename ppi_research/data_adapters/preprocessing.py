from __future__ import annotations
import math
import re
import typing
from itertools import chain


def log_transform_labels(
    label: float,
    base: float = 10,
    eps: float = 1e-10,
) -> float:
    return abs(math.log(label + eps, base))


def insert_sep_token_between_chains(
    sequences: typing.List[str],
    sep_token: str,
    merge_chains: bool = True,
) -> typing.List[str]:
    if not isinstance(sequences[0], list):
        return sequences + [sep_token]

    output = [seq + [sep_token] for seq in sequences]

    if merge_chains:
        output = list(chain.from_iterable(output))
    return output


def prott5_sequence_preprocessing(
    sequences: typing.List[str] | str,
) -> typing.List[str] | str:
    if isinstance(sequences, str):
        return re.sub(r"[UZOB]", "X", sequences)

    output = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
    return output


def prott5_sequence_pair_preprocessing(ligands, receptors):
    def preprocess_sequence(sequence: str) -> typing.List[str]:
        if isinstance(sequence, str):
            sequence = [sequence]
        sequence = prott5_sequence_preprocessing(sequence, sep_token="</s>")
        return sequence

    ligands = preprocess_sequence(ligands)
    receptors = preprocess_sequence(receptors)
    return ligands + receptors


def multi_chain_preprocessing(
    sequences: str,
    sep=",",
) -> typing.List[str]:
    output_seq = sequences.split(sep)
    if len(output_seq) == 1:
        return output_seq[0]
    return output_seq


def split_string_sequences_to_list(
    sequences: str | typing.List[str],
) -> typing.Tuple[typing.List[str] | typing.List[typing.List[str]], int]:
    """Convert a string or a list of strings to
    a list of lists and return the number of chains.

    Args:
        sequences (str | typing.List[str]): The sequences to convert.

    Returns:
        typing.Tuple[typing.List[str] | typing.List[typing.List[str]], int]:
        The converted sequences and the number of chains.
    """
    if isinstance(sequences, str):
        # wrapping this list of characters
        # in another list to make the output
        # of multi chain and single chain
        # consistent (both of them returns a list of lists)
        return [list(sequences)], 1
    output = [list(seq) for seq in sequences]
    return output, len(output)
