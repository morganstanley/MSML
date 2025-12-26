from typing import Callable, Iterable, Optional

from gluonts.dataset import Dataset
from gluonts.dataset.loader import Batch
from gluonts.itertools import IterableSlice, PseudoShuffled
from gluonts.transform import AdhocTransform, Identity
from gluonts.transform._base import Transformation

from .itertools import WithoutReplacementSampled


def NonCyclicTrainDataLoader(
        dataset: Dataset,
        *,
        transform: Transformation = Identity(),
        batch_size: int,
        stack_fn: Callable,
        num_batches_per_epoch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None) -> Iterable:
    """Reimplementation of gluonts loader without making dataset cyclical.

    This train loader optionally shuffles the dataset.
    It then iterates over it, applying transform and creating batches in the process.
    It proceeds until num_batches_per_epoch batches have been yielded.

    Note that this is a function and not a class, due to gluonTS wackyness.

    Args:
        dataset (Dataset): Dataset to load.
        transform (Transformation, optional): Transformation to apply to the time series.
            Defaults to Identity().
        batch_size (int): Number of instances to combine into a batch.
        stack_fn (Callable): Function to use for stacking batches into tensors.
        num_batches_per_epoch (Optional(int)): Number of batches to generate.
        shuffle_buffer-length (Optional(int)): Buffer size for pseudo-shuffling

    Returns:
        IterableSlice: _description_
    """

    # Everything below here is iterable, i.e., we do not instantiate Iterator
    if shuffle_buffer_length:
        dataset = PseudoShuffled(dataset, shuffle_buffer_length)

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    batch_iterable = transform.apply(dataset, is_train=True)

    # We do not call iter(batch_iterable) here
    # So that dataloader can be used multiple times
    return IterableSlice(batch_iterable, num_batches_per_epoch)


def WithoutReplacementTrainDataLoader(
        dataset: Dataset,
        *,
        transform: Transformation = Identity(),
        sample_size: int,
        batch_size: int,
        stack_fn: Callable,
        num_batches_per_epoch: int) -> Iterable:
    """Train loader that uses WOR subsampling for selecting time series.

    This train loader first samples sample_size time series without replacement from dataset.
    It then iterates over them, applying transform and creating batches in the process.
    After sample_size time series have been processed, it samples another sample_size time series.

    Note that this is a function and not a class, due to gluonTS wackyness.

    Args:
        dataset (Dataset): Dataset to load.
        sample_size (int): Number of WOR samples to take simultaneously.
        batch_size (int): Number of instances to combine into a batch.
        stack_fn (Callable): Function to use for stacking batches into tensors.
        transform (Transformation, optional): Transformation to apply to the time series.
            Defaults to Identity().
        num_batches_per_epoch (int): Number of batches to generate.

    Returns:
        IterableSlice: _description_
    """

    # Upper bound to limit the underlying iterator
    max_num_samples = num_batches_per_epoch * batch_size

    dataset = WithoutReplacementSampled(
        dataset, max_num_samples, sample_size)

    transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    batch_iterable = transform.apply(dataset, is_train=True)

    # We do not call iter(batch_iterable) here
    # So that dataloader can be used multiple times
    return IterableSlice(batch_iterable, num_batches_per_epoch)
