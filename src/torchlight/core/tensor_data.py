"""
Dimension layout helpers for tensors: shape / stride arithmetic.

This module is the lowest layer of Torchlight.  It defines:

* Scalar numeric helpers (:func:`prod`).
* Pure-Python index <-> storage-position conversions
  (:func:`index_to_position`, :func:`to_index`, :func:`broadcast_index`).
* NumPy broadcasting rules (:func:`shape_broadcast`).
* The :class:`TensorData` class, which wraps a flat numpy ``float32``
  buffer plus a shape/strides description.  Everything above this layer
  (autograd, backends, ``nn``) talks through :class:`TensorData`.

All TensorData storage is host numpy memory, even for the ``numba`` /
``cuda`` backends -- those backends transfer per-kernel, so no device-aware
buffer type is needed here.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from .device import Device, cpu, resolve_device


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
#: A single 1-D block of numbers (the raw memory of a tensor).
Storage: "type alias" = Any  # np.ndarray
#: An index tuple (e.g. ``(1, 3)``).
UserIndex: "type alias" = Tuple[int, ...]
#: A shape tuple (e.g. ``(2, 3)``).
UserShape: "type alias" = Tuple[int, ...]
#: A strides tuple.
UserStrides: "type alias" = Tuple[int, ...]

#: Maximum number of dimensions a tensor may have.  Kept generous so that
#: internal scratch buffers (index arrays etc.) never overflow.
MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Raised whenever an index/shape/strides inconsistency is found."""


# ---------------------------------------------------------------------------
# Scalar / list helpers
# ---------------------------------------------------------------------------
def prod(shape: Sequence[int]) -> int:
    """Product of the entries of ``shape`` (total number of elements)."""
    out = 1
    for s in shape:
        out *= s
    return out


def strides_from_shape(shape: UserShape) -> UserStrides:
    """
    Produce the row-major (contiguous) strides for a shape.

    For ``shape == (2, 3)`` this is ``(3, 1)``: moving one step along the
    outermost dimension skips 3 elements, moving one step along the innermost
    dimension skips 1 element.
    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


# ---------------------------------------------------------------------------
# Index <-> position conversion
# ---------------------------------------------------------------------------
def index_to_position(index: Sequence[int], strides: Sequence[int]) -> int:
    """
    Convert a multidimensional ``index`` into a flat storage ``position``.

    ``position = sum(index[i] * strides[i])``  -- the standard strided layout.
    """
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position


def to_index(ordinal: int, shape: Sequence[int], out_index: "npt.NDArray[np.int64]") -> None:
    """
    Convert a flat ``ordinal`` (0 .. size-1) into a multidimensional index
    for ``shape``, writing into ``out_index`` in-place.

    Enumerating ordinals 0..size-1 visits every index exactly once.
    """
    cur_ord = ordinal
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Sequence[int],
    big_shape: Sequence[int],
    shape: Sequence[int],
    out_index: "npt.NDArray[np.int64]",
) -> None:
    """
    Map an index in the *bigger* tensor (after broadcasting) back to an index
    in the *smaller* ``shape``, writing into ``out_index`` in-place.

    Extra leading dimensions and size-1 broadcast dimensions map to ``0``.
    """
    for i, s in enumerate(shape):
        if s > 1:
            # Align shapes on the right: big tensor has extra leading dims.
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            # This dimension was stretched by broadcasting -> use 0.
            out_index[i] = 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Compute the shape that results from broadcasting ``shape1`` and ``shape2``
    together, following numpy's rules:

    * Right-align the shapes.
    * A dimension of 1 broadcasts to any size.
    * Missing leading dimensions are treated as 1.

    Raises :class:`IndexingError` if the two shapes are incompatible.
    """
    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
    return tuple(reversed(c_rev))


# ---------------------------------------------------------------------------
# TensorData
# ---------------------------------------------------------------------------
class TensorData:
    """
    The raw data container for a :class:`torchlight.tensor.Tensor`.

    ``TensorData`` owns a single flat 1-D numpy ``float32`` buffer plus a
    shape and a strides tuple.

    Non-contiguous tensors -- the result of ``permute`` or ``broadcast_to``
    -- share the same buffer with different strides.

    All map / zip / reduce / matmul kernels are implemented *on top of*
    ``TensorData`` by the backends, which resolve strided/broadcast
    layouts to their native efficient representation.
    """

    _storage: Storage
    _strides: Tuple[int, ...]
    _shape: Tuple[int, ...]
    _device: Device

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        device: Optional[Device] = None,
    ):
        # --- Validate inputs -------------------------------------------------
        if len(shape) > MAX_DIMS:
            raise IndexingError(f"Shape {shape} has too many dimensions")
        shape = tuple(int(s) for s in shape)

        # --- Resolve device (host numpy is the only supported storage) -------
        if device is None:
            device = cpu
        else:
            device = resolve_device(device)
            if not device.is_cpu:
                raise ValueError(
                    f"TensorData storage is CPU numpy-backed; got {device}. "
                    "Use the 'numba'/'cuda' backends for accelerated execution."
                )
        self._device = device

        # --- Coerce storage to flat float32 ----------------------------------
        self._storage = np.asarray(storage, dtype=np.float32).reshape(-1)

        # --- Strides ---------------------------------------------------------
        if strides is None:
            strides = strides_from_shape(shape)
        strides = tuple(int(s) for s in strides)
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")

        # --- Consistency -----------------------------------------------------
        # Only contiguous constructions own a full-length storage; strided
        # views (broadcast/permute) intentionally share a shorter buffer.
        if strides == strides_from_shape(shape):
            n_elements = prod(shape)
            if len(self._storage) != n_elements:
                raise IndexingError(
                    f"Storage has {len(self._storage)} values but shape {shape} "
                    f"needs {n_elements}."
                )

        self._shape = shape
        self._strides = strides

    # -- Properties -----------------------------------------------------------
    @property
    def shape(self) -> UserShape:
        """Dimension tuple of the tensor."""
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        """Stride tuple describing the memory layout."""
        return self._strides

    @property
    def size(self) -> int:
        """Total number of elements (product of ``shape``)."""
        return prod(self._shape)

    @property
    def dims(self) -> int:
        """Number of dimensions (0 for a scalar)."""
        return len(self._shape)

    @property
    def dtype(self) -> Any:
        """Dtype of the underlying storage (numpy ``float32``)."""
        return self._storage.dtype

    @property
    def device(self) -> Device:
        """Device this TensorData lives on."""
        return self._device

    @property
    def storage(self) -> Storage:
        """Direct access to the flat storage buffer (mutable on CPU)."""
        return self._storage

    # -- Layout helpers -------------------------------------------------------
    def is_contiguous(self) -> bool:
        """
        Return ``True`` when the layout is row-major with exactly the strides
        ``strides_from_shape(shape)``.
        """
        return self._strides == strides_from_shape(self._shape)

    def numpy_view(self) -> "npt.NDArray[np.float32]":
        """
        Interpret the flat storage as an array with ``self.shape`` and
        ``self.strides`` *without copying memory* (a strided view).

        CPU only.  Raises if called on a TensorData created on another device.
        """
        if not self._device.is_cpu:
            raise RuntimeError(
                "numpy_view() is only available on CPU TensorData.  "
                "Use .to_numpy() to get a dense host copy."
            )
        return np.lib.stride_tricks.as_strided(
            self._storage,
            shape=self._shape,
            strides=tuple(s * self._storage.itemsize for s in self._strides),
        )

    def to_numpy(self) -> "npt.NDArray[np.float32]":
        """Return a *dense* (contiguous) numpy copy of the tensor data."""
        if self.is_contiguous():
            return self._storage.reshape(self._shape).copy()
        return self.numpy_view().copy()

    # -- View-producing operations (share storage) -----------------------------
    def permute(self, *order: int) -> "TensorData":
        """
        Reorder the dimensions of the tensor.

        Returns a new :class:`TensorData` sharing the same storage but with the
        shape and strides permuted.  This is how ``transpose`` is built.
        """
        if list(sorted(order)) != list(range(self.dims)):
            raise IndexingError(
                f"Must give a position for each dimension. "
                f"Shape: {self.shape} Order: {order}"
            )
        return TensorData(
            self._storage,
            tuple(self._shape[o] for o in order),
            tuple(self._strides[o] for o in order),
            device=self._device,
        )

    def broadcast_to(self, shape: UserShape) -> "TensorData":
        """
        Return a strided *view* that broadcasts this tensor to ``shape``.

        Broadcast dimensions get stride 0, so they reuse the same storage
        value and never allocate new memory.
        """
        shape = tuple(shape)
        pad = len(shape) - self.dims
        if pad < 0:
            raise IndexingError(f"Cannot broadcast {self.shape} to {shape}")
        padded_shape = (1,) * pad + self._shape
        padded_strides = (0,) * pad + self._strides
        new_strides = []
        for i, dim in enumerate(shape):
            if padded_shape[i] == dim:
                new_strides.append(padded_strides[i])
            elif padded_shape[i] == 1:
                new_strides.append(0)
            else:
                raise IndexingError(f"Broadcast failure {self.shape} {shape}")
        return TensorData(
            self._storage, shape, tuple(new_strides), device=self._device
        )

    def reshape(self, shape: UserShape) -> "TensorData":
        """
        Return a new :class:`TensorData` describing the same elements with a
        new ``shape``.  The storage is reused when the layout is contiguous,
        otherwise the data is materialized into a new contiguous buffer.
        """
        shape = tuple(int(s) for s in shape)
        if prod(shape) != self.size:
            raise IndexingError(f"Cannot reshape {self.shape} -> {shape}")
        if self.is_contiguous():
            return TensorData(self._storage, shape, device=self._device)
        return TensorData(self.to_numpy(), shape, device=self._device)

    def contiguous(self) -> "TensorData":
        """Return a contiguous copy of this tensor (no-op if already so)."""
        if self.is_contiguous():
            return self
        return TensorData(self.to_numpy(), self._shape)

    # -- Allocation helpers ---------------------------------------------------
    @staticmethod
    def zeros(
        shape: UserShape,
        dtype: Any = None,
        device: Optional[Device] = None,
    ) -> "TensorData":
        """Allocate a tensor filled with zeros (numpy float32 host memory)."""
        shape = tuple(shape)
        resolve_device(device)  # validates the device is known
        return TensorData(np.zeros(prod(shape), dtype=np.float32), shape)

    @staticmethod
    def ones(
        shape: UserShape,
        dtype: Any = None,
        device: Optional[Device] = None,
    ) -> "TensorData":
        """Allocate a tensor filled with ones (numpy float32 host memory)."""
        shape = tuple(shape)
        resolve_device(device)  # validates the device is known
        return TensorData(np.ones(prod(shape), dtype=np.float32), shape)

    @staticmethod
    def empty(
        shape: UserShape,
        dtype: Any = None,
        device: Optional[Device] = None,
    ) -> "TensorData":
        """Allocate an uninitialised tensor (caller must fill it)."""
        shape = tuple(shape)
        resolve_device(device)  # validates the device is known
        return TensorData(np.empty(prod(shape), dtype=np.float32), shape)

    # -- Indexing -------------------------------------------------------------
    def _flat_index(self, index: UserIndex) -> int:
        """Validate ``index`` and return the flat storage position."""
        if len(index) != self.dims:
            raise IndexingError(f"Index {index} must be size of {self.shape}.")
        for i, ind in enumerate(index):
            if ind < 0 or ind >= self._shape[i]:
                raise IndexingError(f"Index {index} out of range {self.shape}.")
        return index_to_position(index, self._strides)

    def get(self, key: UserIndex) -> float:
        """Read the value at ``key`` (a full index tuple)."""
        pos = self._flat_index(key)
        return float(self._storage[pos].item())

    def set(self, key: UserIndex, val: float) -> None:
        """Write ``val`` at ``key`` (a full index tuple)."""
        if not self._device.is_cpu:
            raise RuntimeError(
                "In-place set() is only available on CPU TensorData.  "
                "Use backend ops to produce new device arrays."
            )
        self._storage[self._flat_index(key)] = val

    def __getitem__(self, key: Union[int, Tuple[int, ...]]) -> float:
        """``td[i, j, ...]`` reads a single element (full-index only)."""
        if not isinstance(key, tuple):
            key = (key,)
        return self.get(key)

    def __setitem__(self, key: Union[int, Tuple[int, ...]], val: float) -> None:
        """``td[i, j, ...] = v`` writes a single element (full-index only)."""
        if not isinstance(key, tuple):
            key = (key,)
        self.set(key, val)

    def indices(self) -> Iterable[UserIndex]:
        """Yield every full index tuple of this tensor, in row-major order."""
        out_index: "npt.NDArray[np.int64]" = np.empty(self.dims, dtype=np.int64)
        for i in range(self.size):
            to_index(i, self._shape, out_index)
            yield tuple(int(x) for x in out_index)

    def sample(self) -> UserIndex:
        """Return a random valid index tuple (used for gradient checks)."""
        return tuple(random.randint(0, s - 1) for s in self._shape)

    # -- Printing -------------------------------------------------------------
    def __repr__(self) -> str:
        body = " ".join(f"{v:f}" for v in self.to_numpy().reshape(-1))
        return f"TensorData({body} shape={self._shape} strides={self._strides})"


# Convenience re-export so `from .tensor_data import MAX_DIMS` stays intuitive.
__all__ = [
    "Storage",
    "UserIndex",
    "UserShape",
    "UserStrides",
    "MAX_DIMS",
    "IndexingError",
    "prod",
    "strides_from_shape",
    "index_to_position",
    "to_index",
    "broadcast_index",
    "shape_broadcast",
    "TensorData",
]