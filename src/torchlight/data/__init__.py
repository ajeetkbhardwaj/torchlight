"""
torchlight.data -- datasets and data loading utilities.

* :class:`Dataset`        -- abstract indexable dataset.
* :class:`TensorDataset`  -- an in-memory dataset built from tensors/arrays.
* :class:`DataLoader`     -- batches + shuffles a dataset.
"""

from .dataloader import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from .dataset import Dataset, TensorDataset
from .synthetic import Graph, make_spiral, make_synthetic, synthetic_classification

__all__ = [
    "Dataset",
    "TensorDataset",
    "DataLoader",
    "SequentialSampler",
    "RandomSampler",
    "BatchSampler",
    "Graph",
    "make_synthetic",
    "make_spiral",
    "synthetic_classification",
]