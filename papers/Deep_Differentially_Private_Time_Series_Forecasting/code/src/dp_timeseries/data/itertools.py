from dataclasses import dataclass

import numpy as np
from gluonts.itertools import SizedIterable


class WithoutReplacementSampled:
    """A WOR-subsampled iterable.

    An iterator created from this iterable
    first samples sample_size elements of the underlying iterable.
    It then iterates over them.

    Attributes:
        iterable (SizedIterable): The underlying iterable to sample from.
        num_samples (int): Number of WOR sample rounds before end of iterator.
        sample_size (int): Number of elements to sample per WOR sample round.

    Yields:
        _type_: _description_
    """

    def __init__(self,
                 iterable: SizedIterable,
                 num_samples: int, sample_size: int) -> None:

        self.support = list(iterable)
        self.num_samples = num_samples
        self.sample_size = sample_size

    def __iter__(self):

        for _ in range(self.num_samples):
            sample_idxs = np.random.choice(np.arange(len(self.support)),
                                           size=self.sample_size,
                                           replace=False)

            for i in sample_idxs:
                yield self.support[i]

    def __len__(self):
        return self.num_samples * self.sample_size
