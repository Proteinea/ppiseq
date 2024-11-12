from ppi_research.data_adapters.dataset_adapters import PPIDataset
from datasets import load_dataset


def load_skempi_ppi_dataset():
    ds = load_dataset("proteinea/skempi_ppi")
    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=["protein 1 sequence", "protein 2 sequence"],
        label_column_name="affinity (pKd)",
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=["protein 1 sequence", "protein 2 sequence"],
        label_column_name="affinity (pKd)",
    )
    test_ds = PPIDataset(
        ds["test"],
        sequence_column_names=["protein 1 sequence", "protein 2 sequence"],
        label_column_name="affinity (pKd)",
    )
    return train_ds, {"validation": val_ds, "test": test_ds}


def load_ppi_deepdirect_dataset():
    ds = load_dataset("proteinea/ppi_deepdirect")
    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
    )
    test_ds = PPIDataset(
        ds["test"],
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
    )
    ds = load_dataset(
        "proteinea/ppi_deepdirect", data_files="ab-bind_processed.csv"
    )["train"]
    ab_bind_ds = PPIDataset(
        ds,
        sequence_column_names=["Pre_Mut_Seq", "Aft_Mut_Seq"],
        label_column_name="DDG",
    )
    return train_ds, {
        "validation": val_ds,
        "test": test_ds,
        "ab_bind": ab_bind_ds,
    }


def load_peer_ppi_dataset():
    data_files = {
        "train": "train_split.csv",
        "validation": "valid_split.csv",
        "test": "test_split.csv",
    }
    ds = load_dataset("proteinea/peer_ppi_splits", data_files=data_files)

    train_ds = PPIDataset(
        ds["train"],
        sequence_column_names=["graph1", "graph2"],
        label_column_name="interaction",
    )
    val_ds = PPIDataset(
        ds["validation"],
        sequence_column_names=["graph1", "graph2"],
        label_column_name="interaction",
    )
    test_ds = PPIDataset(
        ds["test"],
        sequence_column_names=["graph1", "graph2"],
        label_column_name="interaction",
    )
    return train_ds, {"validation": val_ds, "test": test_ds}


available_datasets = {
    "skempi": load_skempi_ppi_dataset,
    "peer": load_peer_ppi_dataset,
    "deepdirect": load_ppi_deepdirect_dataset,
}


def load_ppi_dataset(identifier):
    return available_datasets[identifier]()
