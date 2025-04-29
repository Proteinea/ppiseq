import typing
from ppi_research.data_adapters import preprocessing


def preprocess_multi_chain_sequences(
    ligands: str,
    receptors: str,
    sep: str = ",",
) -> typing.Tuple[str, str]:
    """Preprocess a pair of multi-chain sequences.

    Args:
        ligands (str): The ligand sequence.
        receptors (str): The receptor sequence.
        sep (str, optional): The separator between the chains. Defaults to ",".

    Returns:
        typing.Tuple[str, str]: The processed ligand and receptor sequences.
    """
    ligands = preprocessing.multi_chain_preprocessing(ligands, sep)
    receptors = preprocessing.multi_chain_preprocessing(receptors, sep)
    return ligands, receptors


def add_sep_tokens_between_chains(
    ligands: str,
    receptors: str,
    sep_token: str,
    merge_chains: bool = True,
) -> typing.Tuple[str, str]:
    """Add a separator token between chains.

    Args:
        ligands (str): The ligand sequence.
        receptors (str): The receptor sequence.
        sep_token (str): The separator token to insert.
        merge_chains (bool, optional): Whether to merge chains.
        Defaults to True.

    Returns:
        typing.Tuple[str, str]: The processed ligand and receptor sequences.
    """
    ligands = preprocessing.insert_sep_token_between_chains(
        ligands,
        sep_token=sep_token,
        merge_chains=merge_chains,
    )
    receptors = preprocessing.insert_sep_token_between_chains(
        receptors,
        sep_token=sep_token,
        merge_chains=merge_chains,
    )
    return ligands, receptors


class SequencePairPreprocessingPipeline:
    def __init__(self, model_name: str):
        """Initialize the preprocessing pipeline.

        Args:
            model_name (str): The name of the model.
        """
        self.model_name = model_name.lower()
        # ESM and ESM3 are handled in the same way
        if self.model_name in ["esm", "esm3"]:
            self._suffix = "<eos>"
            self._preprocessing_function = self._esm_preprocessing
        elif self.model_name == "ankh":
            self._suffix = "</s>"
            self._preprocessing_function = self._ankh_preprocessing
        elif self.model_name == "prott5":
            self._suffix = "</s>"
            self._preprocessing_function = self._prott5_preprocessing
        else:
            raise ValueError(f"Model name {self.model_name} not supported.")

    def _prott5_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> typing.Tuple[str, str]:
        """Preprocess a pair of sequences for ProtT5.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            typing.Tuple[str, str]: The processed ligand and
            receptor sequences.
        """
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )

        ligands = preprocessing.prott5_sequence_preprocessing(ligands)
        receptors = preprocessing.prott5_sequence_preprocessing(receptors)

        ligands, _ = preprocessing.split_string_sequences_to_list(ligands)
        receptors, _ = preprocessing.split_string_sequences_to_list(receptors)

        ligands, receptors = add_sep_tokens_between_chains(
            ligands, receptors, sep_token=self._suffix, merge_chains=True
        )
        return ligands, receptors

    def _ankh_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> typing.Tuple[str, str]:
        """Preprocess a pair of sequences for Ankh.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            typing.Tuple[str, str]: The processed ligand and
            receptor sequences.
        """
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands, _ = preprocessing.split_string_sequences_to_list(ligands)
        receptors, _ = preprocessing.split_string_sequences_to_list(receptors)

        ligands, receptors = add_sep_tokens_between_chains(
            ligands, receptors, sep_token=self._suffix, merge_chains=True
        )
        return ligands, receptors

    def _esm_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> typing.Tuple[str, str]:
        """Preprocess a pair of sequences for ESM.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            typing.Tuple[str, str]: The processed ligand and
            receptor sequences.
        """
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands, _ = preprocessing.split_string_sequences_to_list(ligands)
        receptors, _ = preprocessing.split_string_sequences_to_list(receptors)
        ligands, receptors = add_sep_tokens_between_chains(
            ligands, receptors, sep_token=self._suffix, merge_chains=True
        )
        ligands = ["<cls>"] + ligands
        receptors = ["<cls>"] + receptors
        return ligands, receptors

    def preprocess(
        self, ligands: str, receptors: str
    ) -> typing.Tuple[str, str]:
        return self._preprocessing_function(ligands, receptors)


class SequenceConcatPreprocessingPipeline(SequencePairPreprocessingPipeline):
    def __init__(self, model_name: str):
        """Initialize the concatenation pipeline.

        Args:
            model_name (str): The name of the model.
        """
        super(SequenceConcatPreprocessingPipeline, self).__init__(model_name)

    def _prott5_preprocessing(self, ligands: str, receptors: str) -> str:
        """Preprocess a pair of sequences for ProtT5.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            str: The concatenated ligand and receptor sequences.
        """
        ligands, receptors = super()._prott5_preprocessing(ligands, receptors)
        return ligands + receptors

    def _ankh_preprocessing(self, ligands: str, receptors: str) -> str:
        """Preprocess a pair of sequences for Ankh.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            str: The concatenated ligand and receptor sequences.
        """
        ligands, receptors = super()._ankh_preprocessing(ligands, receptors)
        return ligands + receptors

    def _esm_preprocessing(self, ligands: str, receptors: str) -> str:
        """Preprocess a pair of sequences for ESM.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            str: The concatenated ligand and receptor sequences.
        """
        ligands, receptors = super()._esm_preprocessing(ligands, receptors)
        # remove the cls token from the receptors
        return ligands + receptors[1:]


class MultiChainPreprocessingPipeline:
    def __init__(self, model_name: str):
        """Initialize the multi-chain preprocessing pipeline.

        Args:
            model_name (str): The name of the model.
        """
        self.model_name = model_name.lower()
        if self.model_name == "prott5":
            self._preprocessing_function = self._prott5_preprocessing
        elif self.model_name == "ankh":
            self._preprocessing_function = self._ankh_preprocessing
        # ESM and ESM3 are handled in the same way
        elif self.model_name in ["esm", "esm3"]:
            self._preprocessing_function = self._esm_preprocessing
        else:
            raise ValueError(f"Model name {self.model_name} not supported.")

    def _prott5_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> typing.Tuple[list[str], int, list[str], int]:
        """Preprocess a pair of sequences for ProtT5.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            typing.Tuple[list[str], int, list[str], int]: The processed
            ligand chains, number of ligand chains, receptor chains, and
            number of receptor chains.
        """
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands = preprocessing.prott5_sequence_preprocessing(ligands)
        receptors = preprocessing.prott5_sequence_preprocessing(receptors)

        (
            ligands,
            num_ligand_chains,
        ) = preprocessing.split_string_sequences_to_list(ligands)
        (
            receptors,
            num_receptor_chains,
        ) = preprocessing.split_string_sequences_to_list(receptors)
        return ligands, num_ligand_chains, receptors, num_receptor_chains

    def _ankh_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> typing.Tuple[list[str], int, list[str], int]:
        """Preprocess a pair of sequences for Ankh.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            typing.Tuple[list[str], int, list[str], int]: The processed
            ligand chains, number of ligand chains, receptor chains, and
            number of receptor chains.
        """
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        (
            ligands,
            num_ligand_chains,
        ) = preprocessing.split_string_sequences_to_list(ligands)
        (
            receptors,
            num_receptor_chains,
        ) = preprocessing.split_string_sequences_to_list(receptors)
        return ligands, num_ligand_chains, receptors, num_receptor_chains

    def _esm_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> typing.Tuple[list[str], int, list[str], int]:
        """Preprocess a pair of sequences for ESM.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            typing.Tuple[list[str], int, list[str], int]: The processed
            ligand chains, number of ligand chains, receptor chains, and
            number of receptor chains.
        """
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        (
            ligands,
            num_ligand_chains,
        ) = preprocessing.split_string_sequences_to_list(ligands)
        (
            receptors,
            num_receptor_chains,
        ) = preprocessing.split_string_sequences_to_list(receptors)
        return ligands, num_ligand_chains, receptors, num_receptor_chains

    def preprocess(
        self, ligands: str, receptors: str
    ) -> typing.Tuple[list[str], int, list[str], int]:
        """Preprocess a pair of sequences.

        Args:
            ligands (str): The ligand sequence.
            receptors (str): The receptor sequence.

        Returns:
            typing.Tuple[list[str], int, list[str], int]: The processed
            ligand chains, number of ligand chains, receptor chains, and
            number of receptor chains.
        """
        return self._preprocessing_function(ligands, receptors)
