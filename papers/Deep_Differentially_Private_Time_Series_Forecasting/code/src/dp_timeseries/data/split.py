import numpy as np
from gluonts.dataset import Dataset
from gluonts.dataset.split import OffsetSplitter


def train_val_split(dataset: Dataset,
                    prediction_length: int,
                    num_validation_windows: int = 0) -> tuple[Dataset, None | Dataset]:
    """Splits given dataset into train and validation set.

    For the train set, we remove num_validation_windows of length prediction_length
        from the end of each sequence in dataset.
    For the validation set, we create num_validation_windows different sequences
        per sequence in dataset, with each adding a window of length
        prediction_length to the previous one.

    Args:
        dataset (Dataset): The complete dataset.
        prediction_length (int): Size of the windows to split off.
        num_validation_windows (int, optional): Number of windows to split off.
            If 0, the validation dataset is None.
            Defaults to 0.

    Returns:
        tuple[Dataset, None | Dataset]: Train and validation set.
            Validation set is None if num_validation_windows is 0.
    """

    if num_validation_windows == 0:
        return dataset, None
    elif num_validation_windows < 0:
        raise ValueError('Number of validation windows must be non-negative.')

    overall_prediction_length = prediction_length * num_validation_windows

    train_val_splitter = OffsetSplitter(
        offset=(-1 * overall_prediction_length))

    dataset_train, generator_val = train_val_splitter.split(dataset)

    dataset_val = generator_val.generate_instances(prediction_length,
                                                   num_validation_windows)

    dataset_val = ConcatDataset(dataset_val)

    return dataset_train, dataset_val


class ConcatDataset:
    """Converts tuple iterators from OffsetSplitter into proper datasets.
    """
    def __init__(self, test_pairs, axis=-1) -> None:
        self.test_pairs = test_pairs
        self.axis = axis

    def _concat(self, test_pairs):
        for t1, t2 in test_pairs:
            data = {
                "target": np.concatenate([t1["target"], t2["target"]], axis=self.axis),
                "start": t1["start"],
            }
            if "item_id" in t1.keys():
                data["item_id"] = t1["item_id"]
            if "feat_static_cat" in t1.keys():
                data["feat_static_cat"] = t1["feat_static_cat"]
            yield data

    def __iter__(self):
        yield from self._concat(self.test_pairs)

    def __len__(self):
        return len(self.test_pairs)
