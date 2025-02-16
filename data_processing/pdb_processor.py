import os
import warnings
from typing import Dict, List, Set, Tuple

from Bio import PDB, BiopythonWarning, SeqIO

warnings.simplefilter("ignore", BiopythonWarning)


class PDBProcessor:
    three_to_one_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "SEC": "U",
        "PYL": "O",
    }

    def __init__(
        self,
        pdb_server: str = "http://files.wwpdb.org",
        raw_pdbs_dir: str = "raw_pdbs",
    ):
        self.pdbl = PDB.PDBList(server=pdb_server, verbose=False)
        self.raw_pdbs_dir = raw_pdbs_dir
        self.pdb_parser = PDB.PDBParser(QUIET=True)
        # Create the directory if it doesn't exist
        os.makedirs(self.raw_pdbs_dir, exist_ok=True)

    @staticmethod
    def three_letter_aa_to_one(aa: str) -> str:
        return PDBProcessor.three_to_one_map.get(aa, "X")

    def parse_mutation(self, mutation: str) -> Tuple[str, int, str, str]:
        """Parse a mutation string into its components. The mutation string is
        expected to be in the format 'chain_[wt-aa]pos[mut-aa]'. For example,
        A_E166A is a mutation in chain A, where the amino acid at position 166
        is mutated from E to A.

        Args:
            mutation (str): The mutation string to parse.

        Returns:
            Tuple[str, int, str, str]: A tuple containing the chain ID, the
            mutation position, the wild-type amino acid, and the mutated amino
            acid.
        """
        chain = mutation[0]
        mutation = mutation.split("_")[-1]
        pos = int(mutation[1:-1])
        wt_aa = mutation[0]
        mut_aa = mutation[-1]
        return (chain, pos, wt_aa, mut_aa)

    def download_pdb(self, pdb_id: str) -> str:
        output_file = os.path.join(self.raw_pdbs_dir, f"{pdb_id}.pdb")
        if os.path.exists(output_file):
            return output_file
        path = self.pdbl.retrieve_pdb_file(
            pdb_id, file_format="pdb", pdir=self.raw_pdbs_dir
        )
        os.rename(path, output_file)
        return output_file

    def _get_residue_pos(self, residue: PDB.Residue) -> int:
        return residue.get_id()[1]

    def _get_residue_insertion_code(self, residue: PDB.Residue) -> str:
        return residue.get_id()[-1]

    def _is_atom_residue(self, residue: PDB.Residue) -> bool:
        return residue.get_id()[0] == " "

    def parse_seqs_from_pdb(
        self,
        pdb_id: str,
        pdb_path: str,
        chain_ids: List[str],
        mutations: List[str] | None = None,
    ) -> Dict[str, str]:
        """Parse protein sequences from a PDB file.

        Args:
            pdb_id (str): The ID of the PDB file.
            pdb_path (str): The path to the PDB file.
            chain_ids (List[str]): A list of chain IDs to extract the sequences
            from.
            mutations (List[str] | None, optional): A list of mutations to
            apply to the sequences. The mutations are expected to be in the
            format 'chain_[wt-aa]pos[mut-aa]'. For example, A_E166A is a
            mutation in chain A, where the amino acid at position 166 is
            mutated from E to A. Defaults to None.

        Returns:
            Dict[str, str]: A dictionary where the keys are the chain IDs and
            the values are the sequences of the chains.
        """

        mutations_per_chain = {chain: set() for chain in chain_ids}
        if mutations is not None:
            for mutation in mutations:
                # adding the mutation to the set to avoid duplicates
                chain, mut_pos, wt_aa, mut_aa = self.parse_mutation(mutation)
                mutations_per_chain[chain].add((mut_pos, wt_aa, mut_aa))
        model = self.pdb_parser.get_structure(pdb_id, pdb_path)[0]
        chains = dict()
        for chain in model:
            if chain.id not in chain_ids:
                continue
            elif len(chain.child_list) == 0:
                raise ValueError(
                    f"Chain {chain.id} in PDB {pdb_id} has no residues"
                )
            seq = ""
            last_pos = self._get_residue_pos(chain.child_list[0]) - 1
            for i, residue in enumerate(chain.child_list):
                # Only consider Amino Acids
                if not self._is_atom_residue(residue):
                    continue
                current_pos = self._get_residue_pos(residue)
                insertion_code = self._get_residue_insertion_code(residue)
                if current_pos != last_pos + 1 and insertion_code == " ":
                    # current_pos must be last_pos + 1 if there is no insertion
                    # code. This is either a numbering error or missing
                    # residues, so we need to find out which one it is.
                    if i < len(chain.child_list) - 1:  # not the last residue
                        next_pos = self._get_residue_pos(
                            chain.child_list[i + 1]
                        )
                        if next_pos > current_pos:
                            # we are now sure that there are missing residues
                            # and not a numbering error. For example, if the
                            # residues are 1, 6, 7 we have missing residues at
                            # positions 2-5. But if the residues are
                            # 1, 22, 3, 4 we have a numbering error not 21
                            # missing residues. For missing residues, we add Xs
                            # to the sequence. For numbering errors, we just
                            # proceed normally.
                            seq += "X" * (current_pos - last_pos - 1)
                    else:
                        # this is the last residue in the chain, so we can't
                        # check the next residue. We just add Xs to the
                        # sequence for the missing residues.
                        seq += "X" * (current_pos - last_pos - 1)

                last_pos = current_pos
                residue_letter = self.apply_mutation_if_any(
                    pdb_id, mutations_per_chain, chain, i
                )
                seq += residue_letter
            chains[chain.id] = seq
        # ** SANITY CHECK **#
        if mutations is not None:
            # Check that all mutations were consumed
            for chain, mutations in mutations_per_chain.items():
                if len(mutations) > 0:
                    raise ValueError(
                        f"Mutations {mutations} were not applied to chain "
                        f"{chain} in PDB {pdb_id}"
                    )
        # ****************** #
        return chains

    def apply_mutation_if_any(
        self,
        pdb_id: str,
        mutations_per_chain: Dict[str, Set[Tuple[int, str, str]]],
        chain: PDB.Chain,
        index: int,
    ) -> str:
        """Apply mutations to a residue if any.

        Args:
            pdb_id (str): PDB ID.
            mutations_per_chain (Dict[str, Set[Tuple[int, str, str]]]): A
            dictionary where the keys are the chain IDs and the values are sets
            of tuples containing the mutation position, the wild-type amino
            acid, and the mutated amino acid.
            chain (PDB.Chain): The chain object.
            index (int): The index of the residue in 'chain.child_list'.

        Returns:
            str: The one-letter amino acid code of the residue after applying
            the mutations. If the WT amino acid at the mutation position is not
            the expected one an IndexError is raised. If the mutation is
            already applied, a warning is raised. If no mutations are found for
            this position, the one-letter amino acid is returned.
        """
        current_residue = chain.child_list[index]
        residue_letter = self.three_letter_aa_to_one(current_residue.resname)
        residue_pos = self._get_residue_pos(current_residue)
        residue_insertion_code = self._get_residue_insertion_code(
            current_residue
        )
        consumed_mutations = []
        for mutation in mutations_per_chain[chain.id]:
            mut_pos, wt_aa, mut_aa = mutation
            if mut_pos == residue_pos and residue_insertion_code == " ":
                consumed_mutations.append(mutation)
                if residue_letter != wt_aa:
                    if (
                        index != len(chain.child_list) - 1
                        and self.three_letter_aa_to_one(
                            chain.child_list[index + 1].resname
                        )
                        == wt_aa
                    ):
                        raise IndexError(
                            "Mutation Error: Expected WT AA at position "
                            f"{mut_pos} in chain {chain.id} in PDB {pdb_id} to"
                            f" be {wt_aa}, but found {residue_letter}. The "
                            f"next AA is {wt_aa}, this might be an off-by-one "
                            "error."
                        )
                    elif (
                        index != 0
                        and self.three_letter_aa_to_one(
                            chain.child_list[index - 1].resname
                        )
                        == wt_aa
                    ):
                        raise IndexError(
                            "Mutation Error: Expected WT AA at position "
                            f"{mut_pos} in chain {chain.id} in PDB {pdb_id} to"
                            f" be {wt_aa}, but found {residue_letter}. The "
                            f"previous AA is {wt_aa}, this might be an off-by-"
                            "one error."
                        )
                    elif residue_letter == mut_aa:
                        warnings.warn(
                            f"Mutation of WT AA {wt_aa} to {mut_aa} "
                            f"at position {mut_pos} in chain {chain.id} in PDB"
                            f" {pdb_id} is already applied."
                        )
                    else:
                        raise IndexError(
                            f"Mutation Error: Expected WT AA at position "
                            f"{mut_pos} in chain {chain.id} in PDB {pdb_id} to"
                            f" be {wt_aa}, but found {residue_letter}."
                        )
                residue_letter = mut_aa
        [mutations_per_chain[chain.id].remove(m) for m in consumed_mutations]

        return residue_letter

    def get_seqres_from_pdb(self, pdb_path) -> Dict[str, str]:
        chains = dict()
        with open(pdb_path, "r") as pdb_file:
            for chain in SeqIO.parse(pdb_file, "pdb-seqres"):
                # The chain.id is in the format '[PDB_ID]:[CHAIN_ID]' and we
                # only want the chain ID
                chain_id = chain.id.split(":")[-1]
                chains[chain_id] = str(chain.seq)
        return chains

    def pdb_has_header(self, pdb_id: str, pdb_path: str) -> bool:
        strcut = self.pdb_parser.get_structure(pdb_id, pdb_path)
        return len(strcut.header["idcode"]) > 0

    def _find_subseq(self, seq: str, subseq: str) -> int:
        """Find the index of the occurence of 'subseq' in 'seq'.

        Args:
            seq (str): sequence to search in.
            subseq (str): subsequence to search for.

        Returns:
            int: The index of the occurence of 'subseq' in 'seq'. If 'subseq'
            is not found, -1 is returned. If 'subseq' is found more than once,
            -2 is returned.
        """
        loc = seq.find(subseq)
        if loc == -1:
            return -1
        if seq[loc + 1 :].find(subseq) != -1:
            return -2
        return loc

    def _print_find_warn(self, index: int, pdb_id: str, chain_id: str) -> None:
        if index == -1:
            warnings.warn(
                "Could not find the missing residues for chain "
                f"{chain_id} in PDB {pdb_id} due to a misalignment "
                "between seqeuences from ATOM records and SEQRES. "
                "Skipping adding missing residues for this chain."
            )
        elif index == -2:
            warnings.warn(
                "Could not recover the missing residues for chain "
                f"{chain_id} in PDB {pdb_id} due to multiple ambiguous "
                "matches. Skipping adding missing residues for this chain."
            )
        else:
            raise ValueError(
                f"Unexpected return value from _find_subseq: {index}"
            )

    def _add_missing_residues(
        self,
        pdb_id: str,
        chains: Dict[str, str],
        raw_chains: Dict[str, str],
        raw_seqres_chains: Dict[str, str],
    ) -> Dict[str, str]:
        """Add missing residues, if found, to each chain sequence in 'chains'.

        Args:
            pdb_id (str): ID of the PDB file.
            chains (Dict[str, str]): A dictionary where the keys are the chain
            IDs and the values are the sequences of the chains as loaded from
            the data PDBs. They may contain mutations and they may or may not
            have Xs for the missing residues depending on whether the PDB from
            the data has undergone modifications to the ATOM records or not.
            raw_chains (Dict[str, str]): A dictionary where the keys are the
            chain IDs and the values are the sequences of the. These sequences
            do not contain any mutations that may be present in the chains
            dictionary and they contain Xs for the missing residues. If the PDB
            from the data has header information and sequences in 'chains' have
            no mutations, then 'raw_chains' will be identical to 'chains'.
            raw_seqres_chains (Dict[str, str]): A dictionary where the keys are
            the chain IDs and the values are the sequences of the chains as
            loaded from the seqres records in the PDB file. These sequences
            contain the full sequence, including the missing residues, but they
            have extra residues at the termini that are not present in the ATOM
            records and should not be added to the sequences in 'chains'.

        Returns:
            Dict[str, str]: A dictionary where the keys are the chain IDs and
            the values are the sequences of the chains with the missing
            residues added (if found).
        """
        chains_with_missing_residues = set()
        for chain_id, seq in chains.items():
            if chain_id not in raw_chains:
                # Sometimes downloaded PDB files do not contain all the chains
                # of the PDB from the data or maybe the chain is present with
                # a different ID. Anyway, we can't recover the missing residues
                # for this chain, so we skip it.
                warnings.warn(
                    f"Can't find chain {chain_id} in raw PDB {pdb_id}. "
                    "Skipping adding missing residues for this chain."
                )
                continue
            raw_seq = raw_chains[chain_id]
            raw_seq_wihout_x = raw_seq.replace("X", "")
            seq_without_x = seq.replace("X", "")
            if len(raw_seq) != len(raw_seq_wihout_x):
                # different lengths mean that there are missing residues
                if len(raw_seq_wihout_x) != len(seq_without_x):
                    # When this happens it means that raw_seq is form a PDB
                    # different from the one in the data, and for some reason
                    # the PDB in the data has undergone some modifications that
                    # are not present in the raw PDB. In this case, we can't
                    # recover the missing residues for this chain.
                    warnings.warn(
                        f"Sequence length mismatch for chain {chain_id} in PDB"
                        f" {pdb_id} between the raw PDB and the PDB in the "
                        "data. Skipping adding missing residues for this "
                        "chain."
                    )
                    continue
                chains_with_missing_residues.add(chain_id)

        if len(chains_with_missing_residues) == 0:
            return chains

        for chain_id in chains_with_missing_residues:
            seqres_seq = raw_seqres_chains[chain_id]
            seq = chains[chain_id]
            raw_seq = raw_chains[chain_id]
            reconstructed_seq = ""
            reconstructed_raw_seq = ""
            seq_pos = 0
            raw_seq_pos = 0
            while raw_seq_pos < len(raw_seq):
                num_missing_residues = 0
                while (
                    raw_seq_pos + num_missing_residues < len(raw_seq)
                    and raw_seq[raw_seq_pos + num_missing_residues] == "X"
                ):
                    num_missing_residues += 1
                if num_missing_residues == 0:
                    # no missing residues, just copy the residue from seq
                    # to reconstructed_seq
                    reconstructed_seq += seq[seq_pos]
                    reconstructed_raw_seq += raw_seq[raw_seq_pos]
                    seq_pos += 1
                    raw_seq_pos += 1
                elif raw_seq_pos == 0:
                    # There are missing residues at the beginning of the
                    # sequence. We have to find the next part of the raw
                    # sequence that aligns with the SEQRES sequence to find
                    # the end index of the missing residues in the SEQRES
                    # then using num_missing_residues we can find the
                    # beginning index of the missing residues in the SEQRES
                    # sequence.
                    subseqs = raw_seq.split("X")
                    # subseqs[:num_missing_residues] will contain empty strings
                    # for the missing residues at the beginning of the sequence
                    # and the first non-empty string at index
                    # num_missing_residues will be the subsequence that we want
                    # to  align with the SEQRES sequence.
                    subseq = subseqs[num_missing_residues]

                    # missing_residues_end = seqres_seq.find(subseq)
                    missing_residues_end = self._find_subseq(
                        seq=seqres_seq, subseq=subseq
                    )
                    if missing_residues_end < 0:
                        self._print_find_warn(
                            missing_residues_end, pdb_id, chain_id
                        )
                        reconstructed_seq = seq
                        break
                    missing_residues_begin = (
                        missing_residues_end - num_missing_residues
                    )
                    missing_residues = seqres_seq[
                        missing_residues_begin:missing_residues_end
                    ]
                    reconstructed_seq += missing_residues
                    reconstructed_raw_seq += missing_residues
                    # skip all the Xs we just added
                    raw_seq_pos += num_missing_residues
                else:
                    # we want to find where the reconstructed_raw_seq aligns
                    # with the seqres_seq. Directly after the subsequence, will
                    # be the missing residues.
                    loc = seqres_seq.find(reconstructed_raw_seq)
                    if loc < 0:
                        self._print_find_warn(loc, pdb_id, chain_id)
                        reconstructed_seq = seq
                        break
                    missing_residues_begin = loc + len(reconstructed_raw_seq)
                    missing_residues_end = (
                        missing_residues_begin + num_missing_residues
                    )
                    missing_residues = seqres_seq[
                        missing_residues_begin:missing_residues_end
                    ]
                    reconstructed_seq += missing_residues
                    reconstructed_raw_seq += missing_residues
                    # skip all the Xs we just added
                    raw_seq_pos += num_missing_residues
            else:
                # If the loop completes without breaking we perform a final
                # check.
                if seqres_seq.find(reconstructed_raw_seq) == -1:
                    # Certain misaligment cases are not handled by the previous
                    # logic. Check it here and warn the user if it happens and
                    # skip adding the missing residues for this chain.
                    warnings.warn(
                        "Could not find the missing residues for chain "
                        f"{chain_id} in PDB {pdb_id} due to a misalignment "
                        "between seqeuences from ATOM records and SEQRES. "
                        "Skipping adding missing residues for this chain."
                    )
                    reconstructed_seq = seq
            chains[chain_id] = reconstructed_seq
        return chains

    def recover_missing_residues(
        self,
        pdb_id: str,
        pdb_path: str,
        chains: Dict[str, str],
        is_mutated: bool = False,
    ) -> Dict[str, str]:
        if not self.pdb_has_header(pdb_id, pdb_path):
            # If the PDB file does not have a header, we can't recover the
            # missing residues, so we download the raw PDB file containing the
            # header. Also, usually in the data when the header is missing, the
            # positions in the ATOM records are modified by removing the
            # discontinuous positions. In this case, the sequences in the
            # 'chains_seq' will not have Xs for the missing residues which is
            # essential for this method to work correctly.
            pdb_path = self.download_pdb(pdb_id)
            raw_pdb = True
        else:
            raw_pdb = False

        if raw_pdb or is_mutated:
            # If the PDB file is the raw PDB file or at least one of the chains
            # were mutated, we need to re-parse the sequences from the PDB file
            raw_chains = self.parse_seqs_from_pdb(
                pdb_id=pdb_id,
                pdb_path=pdb_path,
                chain_ids=list(chains.keys()),
                mutations=None,
            )
        else:
            # If the PDB file is not the raw PDB file and none of the chains
            # were mutated, we can use the sequences in 'chains_seq' directly.
            raw_chains = chains
        # These are the chains loaded from the SEQRES records from the raw PDB.
        # They contain the full sequence, including the missing residues, but
        # they have extra residues at the termini that are not present in the
        # ATOM records, so we should not add them.
        raw_seqres_chains = self.get_seqres_from_pdb(pdb_path)
        return self._add_missing_residues(
            pdb_id, chains, raw_chains, raw_seqres_chains
        )

    def pdb_to_processed_seqs(
        self,
        pdb_id: str,
        pdb_path: str,
        chains: List[str],
        mutations: List[str] | None = None,
        recover_missing_residues: bool = True,
        remove_unk_residues: bool = True,
    ) -> Dict[str, str]:
        chains_seqs = self.parse_seqs_from_pdb(
            pdb_id=pdb_id,
            pdb_path=pdb_path,
            chain_ids=chains,
            mutations=mutations,
        )
        if len(chains_seqs) != len(chains):
            missing_chains = set(chains) - set(chains_seqs.keys())
            raise KeyError(
                f"Can't find chain(s) {missing_chains} in PDB {pdb_id}"
            )
        if recover_missing_residues:
            chains_seqs = self.recover_missing_residues(
                pdb_id, pdb_path, chains_seqs, is_mutated=mutations is not None
            )
        if remove_unk_residues:
            chains_seqs = {
                chain: aa_chain.replace("X", "")
                for chain, aa_chain in chains_seqs.items()
            }
        return chains_seqs
