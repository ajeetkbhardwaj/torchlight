"""
``DataLoader``: batching + shuffling over a :class:`Dataset`.

A :class:`DataLoader` is constructed once with a dataset and hyper-parameters,
then iterated: each pass yields the dataset's elements regrouped into
tensor batches.

    >>> loader = DataLoader(ds, batch_size=32, shuffle=True)
    >>> for x_batch, y_batch in loader:
    ...     ...
"""

from __future__ import annotations

import random
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from ..tensor import Tensor, from_numpy
from .dataset import Dataset


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
class Sampler:
    """Base class for index-sequence producers."""

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Yields ``0, 1, ..., n-1`` (no shuffling)."""

    def __init__(self, n: int):
        self.n = n

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n))

    def __len__(self) -> int:
        return self.n


class RandomSampler(Sampler):
    """Yields a random permutation of ``0 .. n-1`` each epoch."""

    def __init__(self, n: int, seed: int | None = None):
        self.n = n
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed)
        order = list(range(self.n))
        rng.shuffle(order)
        return iter(order)

    def __len__(self) -> int:
        return self.n


class BatchSampler:
    """
    Groups raw indices from a base sampler into batches of ``batch_size``
    (the final batch may be smaller).  This is what gives the DataLoader its
    "epoch" structure.
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch: List[int] = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class DataLoader:
    """
    Wraps a :class:`Dataset` with batching and (optional) shuffling.

    Args:
        dataset: the data source.
        batch_size: samples per batch (last batch may be smaller).
        shuffle: reshuffle indices each epoch.
        drop_last: drop the trailing partial batch.
        seed: optional RNG seed for reproducible shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if not isinstance(dataset, Dataset):
            raise TypeError(f"DataLoader expects a Dataset, got {type(dataset)!r}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[Tensor, ...]]:
        base = RandomSampler(len(self.dataset), seed=self.seed) if self.shuffle else SequentialSampler(len(self.dataset))
        batches = BatchSampler(base, self.batch_size, self.drop_last)
        for batch_indices in batches:
            yield self._collate(batch_indices)

    def __len__(self) -> int:
        base = SequentialSampler(len(self.dataset))
        return len(BatchSampler(base, self.batch_size, self.drop_last))

    def _collate(self, indices: Sequence[int]) -> Tuple[Tensor, ...]:
        """Stack the sampled elements into a batch of tensors."""
        items = [self.dataset[i] for i in indices]
        n_fields = len(items[0])
        stacked = []
        for f in range(n_fields):
            arrays = [it[f].to_numpy() for it in items]
            # Every field leads with a (1, ...) dim from __getitem__ -> stack.
            stacked.append(from_numpy(np.stack(arrays).astype(np.float32)))
        return tuple(stacked)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DataLoader(dataset={self.dataset}, batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, drop_last={self.drop_last})"
        )