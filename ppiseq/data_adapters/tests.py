import unittest

from ppiseq.data_adapters.preprocessing_pipelines import \
    MultiChainPreprocessingPipeline
from ppiseq.data_adapters.preprocessing_pipelines import \
    SequenceConcatPreprocessingPipeline
from ppiseq.data_adapters.preprocessing_pipelines import \
    SequencePairPreprocessingPipeline


class TestPreprocessingPipelines(unittest.TestCase):
    def test_prott5_sequence_pair_preprocessing_with_single_chain(self):
        preprocessor = SequencePairPreprocessingPipeline("prott5")
        ligands = "MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA"
        ligands, receptors = preprocessor.preprocess(ligands, receptors)
        self.assertEqual(
            ligands,
            [
                "M",
                "A",
                "L",
                "W",
                "M",
                "R",
                "L",
                "L",
                "P",
                "L",
                "L",
                "A",
                "L",
                "L",
                "A",
                "L",
                "W",
                "G",
                "P",
                "D",
                "P",
                "A",
                "A",
                "A",
                "</s>",
            ],
        )
        self.assertEqual(
            receptors,
            [
                "M",
                "A",
                "L",
                "W",
                "M",
                "R",
                "L",
                "L",
                "P",
                "L",
                "L",
                "A",
                "L",
                "L",
                "A",
                "L",
                "W",
                "G",
                "P",
                "D",
                "P",
                "A",
                "A",
                "A",
                "</s>",
            ],
        )

    def test_prott5_sequence_pair_preprocessing_with_multi_chain(self):
        preprocessor = SequencePairPreprocessingPipeline("prott5")
        ligands = "MALWMRLLPLLALLALWGPDUZOB,MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        ligands, receptors = preprocessor.preprocess(ligands, receptors)
        expected_ligand = (
            list("MALWMRLLPLLALLALWGPDXXXX")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
        )
        expected_receptor = (
            list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
        )
        self.assertEqual(ligands, expected_ligand)
        self.assertEqual(receptors, expected_receptor)

    def test_ankh_sequence_pair_preprocessing_with_single_chain(self):
        preprocessor = SequencePairPreprocessingPipeline("ankh")
        ligands = "MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA"
        ligands, receptors = preprocessor.preprocess(ligands, receptors)
        expected_ligand = list("MALWMRLLPLLALLALWGPDPAAA") + ["</s>"]
        expected_receptor = list("MALWMRLLPLLALLALWGPDPAAA") + ["</s>"]
        self.assertEqual(ligands, expected_ligand)
        self.assertEqual(receptors, expected_receptor)

    def test_ankh_sequence_pair_preprocessing_with_multi_chain(self):
        preprocessor = SequencePairPreprocessingPipeline("ankh")
        ligands = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        expected_ligand = (
            list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
        )
        expected_receptor = (
            list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
        )
        ligands, receptors = preprocessor.preprocess(ligands, receptors)
        self.assertEqual(ligands, expected_ligand)
        self.assertEqual(receptors, expected_receptor)

    def test_esm_sequence_pair_preprocessing_with_single_chain(self):
        preprocessor = SequencePairPreprocessingPipeline("esm")
        ligands = "MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA"
        ligands, receptors = preprocessor.preprocess(ligands, receptors)
        expected_ligand = (
            ["<cls>"] + list("MALWMRLLPLLALLALWGPDPAAA") + ["<eos>"]
        )
        expected_receptor = (
            ["<cls>"] + list("MALWMRLLPLLALLALWGPDPAAA") + ["<eos>"]
        )
        self.assertEqual(ligands, expected_ligand)
        self.assertEqual(receptors, expected_receptor)

    def test_esm_sequence_pair_preprocessing_with_multi_chain(self):
        preprocessor = SequencePairPreprocessingPipeline("esm")
        ligands = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        ligands, receptors = preprocessor.preprocess(ligands, receptors)
        expected_ligand = (
            ["<cls>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
        )
        expected_receptor = (
            ["<cls>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
        )
        self.assertEqual(ligands, expected_ligand)
        self.assertEqual(receptors, expected_receptor)

    def test_prott5_sequence_concat_preprocessing_with_single_chain(self):
        preprocessor = SequenceConcatPreprocessingPipeline("prott5")
        ligands = "MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPRTR"
        sequence = preprocessor.preprocess(ligands, receptors)
        expected = (
            list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPRTR")
            + ["</s>"]
        )
        self.assertEqual(sequence, expected)

    def test_prott5_sequence_concat_preprocessing_with_multi_chain(self):
        preprocessor = SequenceConcatPreprocessingPipeline("prott5")
        ligands = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        sequence = preprocessor.preprocess(ligands, receptors)
        expected = (
            list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
        )
        self.assertEqual(sequence, expected)

    def test_ankh_sequence_concat_preprocessing_with_single_chain(self):
        preprocessor = SequenceConcatPreprocessingPipeline("ankh")
        ligands = "MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA"
        sequence = preprocessor.preprocess(ligands, receptors)
        expected = (
            list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
        )
        self.assertEqual(sequence, expected)

    def test_ankh_sequence_concat_preprocessing_with_multi_chain(self):
        preprocessor = SequenceConcatPreprocessingPipeline("ankh")
        ligands = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        sequence = preprocessor.preprocess(ligands, receptors)
        expected = (
            list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["</s>"]
        )
        self.assertEqual(sequence, expected)

    def test_esm_sequence_concat_preprocessing_with_single_chain(self):
        preprocessor = SequenceConcatPreprocessingPipeline("esm")
        ligands = "MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA"
        sequence = preprocessor.preprocess(ligands, receptors)
        expected = (
            ["<cls>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
        )
        self.assertEqual(sequence, expected)

    def test_esm_sequence_concat_preprocessing_with_multi_chain(self):
        preprocessor = SequenceConcatPreprocessingPipeline("esm")
        ligands = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        receptors = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        sequence = preprocessor.preprocess(ligands, receptors)
        expected = (
            ["<cls>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
            + list("MALWMRLLPLLALLALWGPDPAAA")
            + ["<eos>"]
        )
        self.assertEqual(sequence, expected)

    def test_prott5_multi_chain_preprocessing(self):
        preprocessor = MultiChainPreprocessingPipeline("prott5")
        ligand_sequence = "MALWMRLLPLLALLALWGPDUZOB,MALWMRLLPLLALLALWGPDPAAA"
        receptor_sequence = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDUZOB"
        expected_ligands = list("MALWMRLLPLLALLALWGPDXXXX"), list(
            "MALWMRLLPLLALLALWGPDPAAA"
        )
        expected_receptors = list("MALWMRLLPLLALLALWGPDPAAA"), list(
            "MALWMRLLPLLALLALWGPDXXXX"
        )
        ligands, receptors = preprocessor.preprocess(
            ligand_sequence, receptor_sequence
        )
        for ligand, expected_ligand in zip(ligands, expected_ligands):
            self.assertEqual(ligand, expected_ligand)
        for receptor, expected_receptor in zip(receptors, expected_receptors):
            self.assertEqual(receptor, expected_receptor)

    def test_ankh_multi_chain_preprocessing(self):
        preprocessor = MultiChainPreprocessingPipeline("ankh")
        ligand_sequence = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        receptor_sequence = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        expected_ligands = list("MALWMRLLPLLALLALWGPDPAAA"), list(
            "MALWMRLLPLLALLALWGPDPAAA"
        )
        expected_receptors = list("MALWMRLLPLLALLALWGPDPAAA"), list(
            "MALWMRLLPLLALLALWGPDPAAA"
        )
        ligands, receptors = preprocessor.preprocess(
            ligand_sequence, receptor_sequence
        )
        for ligand, expected_ligand in zip(ligands, expected_ligands):
            self.assertEqual(ligand, expected_ligand)
        for receptor, expected_receptor in zip(receptors, expected_receptors):
            self.assertEqual(receptor, expected_receptor)

    def test_esm_multi_chain_preprocessing(self):
        preprocessor = MultiChainPreprocessingPipeline("esm")
        ligand_sequence = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        receptor_sequence = "MALWMRLLPLLALLALWGPDPAAA,MALWMRLLPLLALLALWGPDPAAA"
        expected_ligands = list("MALWMRLLPLLALLALWGPDPAAA"), list(
            "MALWMRLLPLLALLALWGPDPAAA"
        )
        expected_receptors = list("MALWMRLLPLLALLALWGPDPAAA"), list(
            "MALWMRLLPLLALLALWGPDPAAA"
        )
        ligands, receptors = preprocessor.preprocess(
            ligand_sequence, receptor_sequence
        )
        for ligand, expected_ligand in zip(ligands, expected_ligands):
            self.assertEqual(ligand, expected_ligand)
        for receptor, expected_receptor in zip(receptors, expected_receptors):
            self.assertEqual(receptor, expected_receptor)


if __name__ == "__main__":
    unittest.main()
