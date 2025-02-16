from ppi_research.data_adapters import preprocessing


def preprocess_multi_chain_sequences(
    ligands: str, receptors: str
) -> tuple[str, str]:
    ligands = preprocessing.multi_chain_preprocessing(ligands)
    receptors = preprocessing.multi_chain_preprocessing(receptors)
    return ligands, receptors


def add_sep_tokens_between_chains(
    ligands: str,
    receptors: str,
    sep_token: str,
    merge_chains: bool = True,
) -> tuple[str, str]:
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
        self.model_name = model_name.lower()
        if self.model_name == "esm":
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
    ) -> tuple[str, str]:
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands = preprocessing.prott5_sequence_preprocessing(ligands)
        receptors = preprocessing.prott5_sequence_preprocessing(receptors)
        ligands, receptors = add_sep_tokens_between_chains(
            ligands, receptors, sep_token=self._suffix, merge_chains=True
        )
        return ligands, receptors

    def _ankh_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> tuple[str, str]:
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands = preprocessing.convert_string_sequences_to_list(ligands)
        receptors = preprocessing.convert_string_sequences_to_list(receptors)
        ligands, receptors = add_sep_tokens_between_chains(
            ligands, receptors, sep_token=self._suffix, merge_chains=True
        )
        return ligands, receptors

    def _esm_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> tuple[str, str]:
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands = preprocessing.convert_string_sequences_to_list(ligands)
        receptors = preprocessing.convert_string_sequences_to_list(receptors)
        ligands, receptors = add_sep_tokens_between_chains(
            ligands, receptors, sep_token=self._suffix, merge_chains=True
        )
        ligands = ["<cls>"] + ligands
        receptors = ["<cls>"] + receptors
        return ligands, receptors

    def preprocess(self, ligands: str, receptors: str) -> tuple[str, str]:
        return self._preprocessing_function(ligands, receptors)


class SequenceConcatPreprocessingPipeline(SequencePairPreprocessingPipeline):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def _prott5_preprocessing(self, ligands: str, receptors: str) -> str:
        ligands, receptors = super()._prott5_preprocessing(ligands, receptors)
        return ligands + receptors

    def _ankh_preprocessing(self, ligands: str, receptors: str) -> str:
        ligands, receptors = super()._ankh_preprocessing(ligands, receptors)
        return ligands + receptors

    def _esm_preprocessing(self, ligands: str, receptors: str) -> str:
        ligands, receptors = super()._esm_preprocessing(ligands, receptors)
        # remove the cls token from the receptors
        return ligands + receptors[1:]


class MultiChainPreprocessingPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        if self.model_name == "prott5":
            self._preprocessing_function = self._prott5_preprocessing
        elif self.model_name == "ankh":
            self._preprocessing_function = self._ankh_preprocessing
        elif self.model_name == "esm":
            self._preprocessing_function = self._esm_preprocessing
        else:
            raise ValueError(f"Model name {self.model_name} not supported.")

    def _prott5_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> tuple[list[str], list[str]]:
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands = preprocessing.prott5_sequence_preprocessing(ligands)
        receptors = preprocessing.prott5_sequence_preprocessing(receptors)
        return ligands, receptors

    def _ankh_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> tuple[list[str], list[str]]:
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands = preprocessing.convert_string_sequences_to_list(ligands)
        receptors = preprocessing.convert_string_sequences_to_list(receptors)
        return ligands, receptors

    def _esm_preprocessing(
        self,
        ligands: str,
        receptors: str,
    ) -> tuple[list[str], list[str]]:
        ligands, receptors = preprocess_multi_chain_sequences(
            ligands, receptors
        )
        ligands = preprocessing.convert_string_sequences_to_list(ligands)
        receptors = preprocessing.convert_string_sequences_to_list(receptors)
        return ligands, receptors

    def preprocess(
        self, ligands: str, receptors: str
    ) -> tuple[list[str], list[str]]:
        return self._preprocessing_function(ligands, receptors)
