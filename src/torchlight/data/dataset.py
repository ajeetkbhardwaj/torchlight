"""
``Dataset`` / ``TensorDataset``: indexable data containers.

A dataset answers two questions: ``len(dataset)`` and ``dataset[i]``.  The
:class:`DataLoader` then shuffles/iterates over these indices to build
batches.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Union

import numpy as np

from ..tensor import Tensor, from_numpy


class Dataset:
    """Abstract base class: implement ``__len__`` and ``__getitem__``."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(len={len(self)})"


class TensorDataset(Dataset):
    """
    A fixed-size dataset built from parallel arrays (features + targets).

    The arrays are converted to Torchlight tensors lazily, one element at a
    time, so the memory stays lightweight for the typical small-scale usage.

    Args:
        *tensors: equally sized array-likes or tensors, one per field.

    Example:
        >>> ds = TensorDataset(tensor([[1.], [2.]]), tensor([0, 1]))
        >>> ds[0]
        (tensor(...), tensor(...))
    """

    def __init__(self, *tensors: Union[Tensor, np.ndarray, Sequence[Any]]):
        self.tensors = [_as_tensor(t) for t in tensors]
        if not self.tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        lengths = [t.shape[0] for t in self.tensors]
        if len(set(lengths)) != 1:
            raise ValueError(f"All tensors must share the first dim, got sizes {lengths}")
        self._len = lengths[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        if idx < 0 or idx >= self._len:
            raise IndexError(f"index {idx} out of range for length {self._len}")
        out = []
        for t in self.tensors:
            arr = t.to_numpy()[idx]
            if arr.ndim == 0:
                # Scalars are kept as shape-(1,) tensors (0-dim would leak
                # indexing special cases everywhere downstream).
                out.append(_as_tensor(np.asarray([arr])))
            else:
                out.append(from_numpy(np.ascontiguousarray(arr, dtype=np.float32)))
        return tuple(out)

    def to_batch(self) -> Tuple[Tensor, ...]:
        """Return the whole dataset as a tuple of (N, ...) tensors."""
        return tuple(t.contiguous() for t in self.tensors)


def _as_tensor(x: Union[Tensor, np.ndarray, Sequence[Any]]) -> Tensor:
    """Coerce the supported input types into a contiguous Torchlight tensor."""
    if isinstance(x, Tensor):
        return x.contiguous()
    arr = np.asarray(x)
    if arr.dtype.kind not in ("f"):
        arr = arr.astype(np.float32)  # integer/bool targets become float tensors
    return from_numpy(np.ascontiguousarray(arr, dtype=np.float32))