import unittest
import warnings
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

from Bio.PDB import Chain, Model, Residue

from pdb_processor import PDBProcessor


class TestAddMissingResidues(unittest.TestCase):
    def setUp(self):
        self.processor = PDBProcessor()
        self.pdb_id = "1TEST"

    def tearDown(self):
        warnings.resetwarnings()

    # ------------------------------------------
    # Core Functionality Tests
    # ------------------------------------------

    def test_no_missing_residues(self):
        chains = {"A": "ABCDEF"}
        result = self.processor._add_missing_residues(
            self.pdb_id,
            chains,
            raw_chains={"A": "ABCDEF"},
            raw_seqres_chains={"A": "ABCDEF"},
        )
        self.assertEqual(result, {"A": "ABCDEF"})

    def test_single_missing_residue_beginning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"A": "BCDEF"},
                raw_chains={"A": "XBCDEF"},
                raw_seqres_chains={"A": "ABCDEF"},
            )
        self.assertEqual(result["A"], "ABCDEF")

    def test_single_missing_residue_middle(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"A": "ABDEF"},
                raw_chains={"A": "ABXDEF"},
                raw_seqres_chains={"A": "ABCDEF"},
            )
        self.assertEqual(result["A"], "ABCDEF")

    def test_single_missing_residue_end(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"A": "ABCDE"},
                raw_chains={"A": "ABCDEX"},
                raw_seqres_chains={"A": "ABCDEF"},
            )
        self.assertEqual(result["A"], "ABCDEF")

    # ------------------------------------------
    # Boundary Cases
    # ------------------------------------------

    # def test_all_x_raw_chain(self):
    #     with self.assertWarns(UserWarning) as cm:
    #         result = self.processor._add_missing_residues(
    #             self.pdb_id,
    #             chains={"A": ""},
    #             raw_chains={"A": "XXXXXX"},
    #             raw_seqres_chains={"A": "ABCDEF"},
    #         )
    #     self.assertIn("Could not find the missing residues", str(cm.warning))
    #     self.assertEqual(result["A"], "")

    # def test_single_residue_chain(self):
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("error")
    #         result = self.processor._add_missing_residues(
    #             self.pdb_id,
    #             chains={"A": "B"},
    #             raw_chains={"A": "XB"},
    #             raw_seqres_chains={"A": "AB"},
    #         )
    #     self.assertEqual(result["A"], "AB")

    # ------------------------------------------
    # Error Condition Tests
    # ------------------------------------------

    def test_chain_mismatch(self):
        with self.assertWarns(UserWarning) as cm:
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"B": "ABCD"},
                raw_chains={"A": "XXXX"},
                raw_seqres_chains={"A": "ABCD"},
            )
        self.assertIn("Can't find chain B", str(cm.warning))
        self.assertEqual(result, {"B": "ABCD"})

    def test_length_mismatch_warning(self):
        with self.assertWarns(UserWarning) as cm:
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"A": "ABCDE"},
                raw_chains={"A": "AXXD"},
                raw_seqres_chains={"A": "ABCDE"},
            )
        self.assertIn("Sequence length mismatch", str(cm.warning))
        self.assertEqual(result["A"], "ABCDE")

    def test_sequence_misalignment1(self):
        # misalignment where the logic will work but the final sequence will
        # be wrong. Should be caught by the final sequence validation in
        # PDBProcessor._add_missing_residues
        with self.assertWarns(UserWarning) as cm:
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"A": "ABCD"},
                raw_chains={"A": "ABXXDE"},
                raw_seqres_chains={"A": "ABCDE"},
            )
        self.assertIn("due to a misalignment between", str(cm.warning))
        self.assertEqual(result["A"], "ABCD")

    def test_sequence_misalignment2(self):
        # misalignment where the logic will fail since ACB will not be found
        # in the raw_seqres_chains.
        with self.assertWarns(UserWarning) as cm:
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"A": "ACB"},
                raw_chains={"A": "ACBX"},
                raw_seqres_chains={"A": "ABCD"},
            )
        self.assertIn("due to a misalignment between", str(cm.warning))
        self.assertEqual(result["A"], "ACB")

    # ------------------------------------------
    # Complex Scenarios
    # ------------------------------------------

    def test_multiple_missing_blocks(self):
        result = self.processor._add_missing_residues(
            self.pdb_id,
            chains={"A": "CDEGIJ"},
            raw_chains={"A": "XXCDEXGXIJX"},
            raw_seqres_chains={"A": "ABCDEFGHIJK"},
        )
        self.assertEqual(result["A"], "ABCDEFGHIJK")

    def test_overlapping_subsequences(self):
        # a case with almost zero chance of happening in real life due to the
        # number of residues in the sequence, but we need to test that the
        # function will detect that something is wrong and avoid modifying the
        # sequence.
        with self.assertWarns(UserWarning) as cm:
            # here we assume the correct reconstruction would be "FGCDEHL" but
            # because of the overlapping subsequences, it is ambiguous to the
            # function.
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"A": "CDE"},
                raw_chains={"A": "XXCDEXX"},
                raw_seqres_chains={"A": "ABCDEFGCDEHL"},
            )
        self.assertIn("due to multiple ambiguous matches", str(cm.warning))
        self.assertEqual(result["A"], "CDE")

    # ------------------------------------------
    # Parameterized-style Tests
    # ------------------------------------------

    def test_parameterized_valid_cases(self):
        test_cases = [
            ("XXEFG", "ABCDEFGHI", "CDEFG"),
            ("CDXXG", "ABCDEFGHI", "CDEFG"),
            ("CDEXX", "ABCDEFGHI", "CDEFG"),
            ("XDEFX", "ABCDEFGHI", "CDEFG"),
            ("XXEFX", "ABCDEFGHI", "CDEFG"),
            ("XDEXX", "ABCDEFGHI", "CDEFG"),
            ("XXEXX", "ABCDEFGHI", "CDEFG"),
            ("CXEXG", "ABCDEFGHI", "CDEFG"),
            ("CXXXG", "ABCDEFGHI", "CDEFG"),
        ]

        for raw_seq, seqres, expected in test_cases:
            with self.subTest(raw_seq=raw_seq, seqres=seqres):
                chains = {"A": raw_seq.replace("X", "")}
                result = self.processor._add_missing_residues(
                    self.pdb_id, chains, {"A": raw_seq}, {"A": seqres}
                )
                self.assertEqual(result["A"], expected)

    def test_parameterized_valid_cases_with_mutation(self):
        # Mutations: B -> A, H -> L
        test_cases = [
            ("BXXEFGH", "ABCDEFGHI", "ACDEFGL"),
            ("BCDXXGH", "ABCDEFGHI", "ACDEFGL"),
            ("BCDEXXH", "ABCDEFGHI", "ACDEFGL"),
            ("BXDEFXH", "ABCDEFGHI", "ACDEFGL"),
            ("BXXEFXH", "ABCDEFGHI", "ACDEFGL"),
            ("BXDEXXH", "ABCDEFGHI", "ACDEFGL"),
            ("BXXEXXH", "ABCDEFGHI", "ACDEFGL"),
            ("BCXEXGH", "ABCDEFGHI", "ACDEFGL"),
            ("BCXXXGH", "ABCDEFGHI", "ACDEFGL"),
        ]

        for raw_seq, seqres, expected in test_cases:
            with self.subTest(raw_seq=raw_seq, seqres=seqres):
                chains = {
                    "A": raw_seq.replace("X", "")
                    .replace("B", "A")
                    .replace("H", "L")
                }
                result = self.processor._add_missing_residues(
                    self.pdb_id, chains, {"A": raw_seq}, {"A": seqres}
                )
                self.assertEqual(result["A"], expected)

    # ------------------------------------------
    # Special Case Tests
    # ------------------------------------------

    def test_mixed_case_chains(self):
        with self.assertWarns(UserWarning):
            result = self.processor._add_missing_residues(
                self.pdb_id,
                chains={"a": "DEF"},
                raw_chains={"A": "XXDEF"},
                raw_seqres_chains={"A": "ABCDEF"},
            )
        self.assertEqual(result["a"], "DEF")

    def test_large_sequence_reconstruction(self):
        result = self.processor._add_missing_residues(
            self.pdb_id,
            chains={"A": "HIJKLQRS"},
            raw_chains={"A": "XXHIJKLXXXXQRSXX"},
            raw_seqres_chains={"A": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        )
        self.assertEqual(result["A"], "FGHIJKLMNOPQRSTU")


class TestParseSeqsFromPDB(unittest.TestCase):
    def setUp(self):
        self.parser = PDBProcessor()
        self.parser.pdb_parser = MagicMock()
        self.parser.three_letter_aa_to_one = MagicMock(
            side_effect=lambda x: {"ALA": "A", "GLU": "E", "VAL": "V"}.get(
                x, "X"
            )
        )
        self.parser.parse_mutation = MagicMock()

    def create_mock_residue(
        self,
        pos: int,
        resname: str,
        insertion_code: str = " ",
        hetero_flag: str = " ",
    ):
        residue = MagicMock(spec=Residue.Residue)
        residue.get_id.return_value = (hetero_flag, pos, insertion_code)
        residue.resname = resname
        return residue

    def create_mock_chain(self, chain_id: str, residues: List[MagicMock]):
        chain = MagicMock(spec=Chain.Chain)
        chain.id = chain_id
        chain.child_list = residues
        return chain

    def test_single_chain_no_mutations(self):
        # Setup: Single chain with residues 1 (A), 2 (V), 3 (E)
        residues = [
            self.create_mock_residue(1, "ALA"),
            self.create_mock_residue(2, "VAL"),
            self.create_mock_residue(3, "GLU"),
        ]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]

        result = self.parser.parse_seqs_from_pdb("test", "path", ["A"])
        self.assertEqual(result, {"A": "AVE"})

    def test_missing_residues_with_x(self):
        # Residues 1 (A), 3 (E) with next residue at 4 (V)
        residues = [
            self.create_mock_residue(1, "ALA"),
            self.create_mock_residue(3, "GLU"),
            self.create_mock_residue(4, "VAL"),
        ]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]

        result = self.parser.parse_seqs_from_pdb("test", "path", ["A"])
        self.assertEqual(result, {"A": "AXEV"})

    def test_multiple_missing_residues_with_x(self):
        # Residues 1 (A), 3 (E) with next residue at 4 (V)
        residues = [
            self.create_mock_residue(1, "ALA"),
            self.create_mock_residue(6, "GLU"),
            self.create_mock_residue(7, "VAL"),
        ]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]

        result = self.parser.parse_seqs_from_pdb("test", "path", ["A"])
        self.assertEqual(result, {"A": "AXXXXEV"})

    def test_numbering_error_no_x(self):
        residues = [
            self.create_mock_residue(1, "ALA"),
            self.create_mock_residue(11, "GLU"),
            self.create_mock_residue(2, "VAL"),
        ]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]

        result = self.parser.parse_seqs_from_pdb("test", "path", ["A"])
        self.assertEqual(result, {"A": "AEV"})

    def test_mutation_applied(self):
        # Residue 166 (E), mutation to A
        residues = [self.create_mock_residue(166, "GLU")]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]
        self.parser.parse_mutation.return_value = ("A", 166, "E", "A")

        result = self.parser.parse_seqs_from_pdb(
            "test", "path", ["A"], ["A_E166A"]
        )
        self.assertEqual(result["A"], "A")

    def test_mutation_wrong_wild_type(self):
        # Residue 166 (A), mutation expects E
        residues = [self.create_mock_residue(166, "ALA")]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]
        self.parser.parse_mutation.return_value = ("A", 166, "E", "R")

        with self.assertRaises(IndexError):
            self.parser.parse_seqs_from_pdb("test", "path", ["A"], ["A_E166R"])

    def test_mutation_already_applied_warning(self):
        # Residue 166 (A), mutation A to A (redundant)
        residues = [self.create_mock_residue(166, "ALA")]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]
        self.parser.parse_mutation.return_value = ("A", 166, "E", "A")

        with self.assertWarns(UserWarning):
            result = self.parser.parse_seqs_from_pdb(
                "test", "path", ["A"], ["A_E166A"]
            )
        self.assertEqual(result["A"], "A")

    def test_mutation_insertion_code_not_applied(self):
        # Residue 166 has insertion code 'A'
        residues = [self.create_mock_residue(166, "GLU", insertion_code="A")]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]
        self.parser.parse_mutation.return_value = ("A", 166, "E", "A")

        with self.assertRaises(ValueError):
            self.parser.parse_seqs_from_pdb("test", "path", ["A"], ["A_E166A"])

    def test_hetero_residue_skipped(self):
        # Hetero residue (H_ALA) is skipped
        residues = [self.create_mock_residue(1, "ALA", hetero_flag="H")]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]

        result = self.parser.parse_seqs_from_pdb("test", "path", ["A"])
        self.assertEqual(result["A"], "")

    def test_sanity_check_unapplied_mutations(self):
        # Mutation to position not present
        residues = [self.create_mock_residue(1, "GLU")]
        chain = self.create_mock_chain("A", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain])
        self.parser.pdb_parser.get_structure.return_value = [model]
        self.parser.parse_mutation.return_value = ("A", 166, "C", "Q")

        with self.assertRaises(ValueError):
            self.parser.parse_seqs_from_pdb("test", "path", ["A"], ["A_C166Q"])

    def test_multiple_chains(self):
        # Chains A and B with residues
        residues_a = [self.create_mock_residue(1, "ALA")]
        residues_b = [self.create_mock_residue(1, "VAL")]
        chain_a = self.create_mock_chain("A", residues_a)
        chain_b = self.create_mock_chain("B", residues_b)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain_a, chain_b])
        self.parser.pdb_parser.get_structure.return_value = [model]

        result = self.parser.parse_seqs_from_pdb("test", "path", ["A", "B"])
        self.assertEqual(result, {"A": "A", "B": "V"})

    def test_chain_not_in_list_ignored(self):
        # Chain C exists but not in chain_ids
        residues = [self.create_mock_residue(1, "ALA")]
        chain_c = self.create_mock_chain("C", residues)
        model = MagicMock(spec=Model.Model)
        model.__iter__.return_value = iter([chain_c])
        self.parser.pdb_parser.get_structure.return_value = [model]

        result = self.parser.parse_seqs_from_pdb("test", "path", ["A"])
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
