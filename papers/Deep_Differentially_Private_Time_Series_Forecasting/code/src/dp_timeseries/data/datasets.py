from pathlib import PosixPath

from gluonts.dataset.common import Dataset, MetaData
from gluonts.dataset.repository import get_dataset


def get_split_dataset(
        dataset_name: str, dataset_dir: str,
        regenerate: bool = False) -> tuple[Dataset, Dataset, MetaData]:
    """Loads gluonts dataset, separates it into default train, test, and meta data.

    Args:
        dataset_name (str): Name of dataset, e.g. "electricity"
        dataset_dir (str): Directory for downloading/loading dataset.
            A new subdirectory will be created within this directory, e.g., "electricity"
        regenerate (bool, optional): Whether to regenerate the dataset
            even if a local file is present.
            Defaults to False.

    Returns:
        tuple[Dataset, Dataset, MetaData]: Train, validation, and meta data.
    """
    dataset = get_dataset(
        dataset_name,
        PosixPath(dataset_dir),
        regenerate=regenerate)

    return dataset.train, dataset.test, dataset.metadata
