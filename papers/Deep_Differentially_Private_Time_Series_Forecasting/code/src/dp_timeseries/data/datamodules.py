from copy import deepcopy

from gluonts.dataset.common import Dataset, MetaData
from gluonts.transform._base import Transformation
from lightning.pytorch.core import LightningDataModule

from .datasets import get_split_dataset
from .split import train_val_split


class GluonTSLightningDataModule(LightningDataModule):
    """Module that abstracts GluonTS Dataloader generation.

    See https://lightning.ai/docs/pytorch/stable/data/datamodule.html.
    """

    def __init__(self,
                 dataset_kwargs: dict) -> None:
        """
        Args:
            dataset_kwargs (dict): Arguments for dataset loading.
                See ./datasets.get_split_dataset.
        """
        super().__init__()

        self.dataset_train: Dataset
        self.dataset_val: Dataset
        self.dataset_val_pred: Dataset  # Seperate dataset for computing PytorchPredictor metrics
        self.dataset_test: Dataset
        self.meta: MetaData

        self.dataset_kwargs = dataset_kwargs

    def prepare_data(self) -> None:
        get_split_dataset(**self.dataset_kwargs)

    def setup(self, stage: str) -> None:

        # Load
        self.dataset_train, self.dataset_test, self.meta = get_split_dataset(
            **self.dataset_kwargs)

        # Split train into train and val
        num_validation_windows = len(self.dataset_test) // len(self.dataset_train)

        self.dataset_train, self.dataset_val = train_val_split(
            self.dataset_train,
            self.meta.prediction_length, num_validation_windows)

        self.dataset_val_pred = deepcopy(self.dataset_val)
