import re


def sequence_preprocessing(sequence):
    return list(re.sub(r"[UZOB]", "X", sequence))


def sequence_pair_preprocessing(inputs):
    seq_1, seq_2 = inputs
    seq_1 = list(re.sub(r"[UZOB]", "X", seq_1))
    seq_2 = list(re.sub(r"[UZOB]", "X", seq_2))
    return seq_1 + ["</s>"] + seq_2
