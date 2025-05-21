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
    """Log transform a label.

    Args:
        label (float): The label to transform.
        base (float, optional): The base of the logarithm. Defaults to 10.
        eps (float, optional): The epsilon value. Defaults to 1e-10.

    Returns:
        float: The transformed label.
    """
    return abs(math.log(label + eps, base))


def insert_sep_token_between_chains(
    sequences: typing.List[str],
    sep_token: str,
    merge_chains: bool = True,
) -> typing.List[str]:
    """Insert a separator token between chains.

    Args:
        sequences (typing.List[str]): The sequences to process.
        sep_token (str): The separator token to insert.
        merge_chains (bool, optional): Whether to merge chains.
        Defaults to True.

    Returns:
        typing.List[str]: The processed sequences.
    """
    if not isinstance(sequences[0], list):
        return sequences + [sep_token]

    output = [seq + [sep_token] for seq in sequences]

    if merge_chains:
        output = list(chain.from_iterable(output))
    return output


def prott5_sequence_preprocessing(
    sequences: typing.List[str] | str,
) -> typing.List[str] | str:
    """Preprocess a sequence for ProtT5.

    Args:
        sequences (typing.List[str] | str): The sequence to preprocess.

    Returns:
        typing.List[str] | str: The processed sequence.
    """
    if isinstance(sequences, str):
        return re.sub(r"[UZOB]", "X", sequences)

    output = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
    return output


def prott5_sequence_pair_preprocessing(
    ligands: str,
    receptors: str,
) -> typing.List[str]:
    """Preprocess a pair of sequences for ProTT5.

    Args:
        ligands (str): The ligand sequence.
        receptors (str): The receptor sequence.

    Returns:
        typing.List[str]: The processed sequences.
    """
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
    sep: str = ",",
) -> typing.List[str]:
    """Split a string of sequences separated by a separator
    into a list of sequences.

    Args:
        sequences (str): The string of sequences to split.
        sep (str, optional): The separator between the sequences.
        Defaults to ",".

    Returns:
        typing.List[str]: The list of sequences.
    """
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
