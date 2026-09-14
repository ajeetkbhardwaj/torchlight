"""
The ``Tensor``: Torchlight's core user-facing object.

A ``Tensor`` wraps two things:

1. a :class:`~torchlight.core.tensor_data.TensorData` -- the raw numpy
   storage + shape/strides, and
2. a :class:`~torchlight.autograd.autodiff.History` -- ``None`` for
   constants, empty for user leaves, or "built by this Function" otherwise.

All operators (``+``, ``*``, ``@``, ``sum``, ``sigmoid`` ...) are thin
delegates to the corresponding :class:`Function` from
:mod:`torchlight.autograd.functions`; ``backward`` walks the resulting graph.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ..autograd.autodiff import Context, History, backpropagate, central_difference
from ..autograd.functions import (
    Abs,
    Add,
    All,
    AttnSoftmax,
    EQ,
    GT,
    Gather,
    IsClose,
    LT,
    LayerNorm,
    MatMul,
    Max,
    Min,
    Mul,
    Neg,
    Permute,
    PowerScalar,
    ReLU,
    Sigmoid,
    Sqrt,
    Sum,
    Tanh,
    View,
    Contiguous,
    Exp,
    Inv,
    Function,
    Log,
)
from ..autograd.indexing import Cat, IndexSelect
from ..backends import CPUBackend, default_backend, get_backend
from ..core.tensor_data import TensorData, prod, shape_broadcast

# Numeric types that may be silently promoted to 1-element tensors.
TensorLike = Union["Tensor", float, int, np.number, Sequence[float], np.ndarray]


#: Global counter used to mint unique ids (stable graph-node identity).
_tensor_count = 0


class Tensor:
    """
    A multidimensional array supporting automatic differentiation on the CPU.

    Attributes:
        _tensor: raw storage + shape/strides.
        backend: the (CPU) backend that executes element-wise kernels.
        history: how this tensor was built (``None`` = constant).
        grad: gradient accumulated by the last :meth:`backward`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        data: TensorData,
        history: Optional[History] = None,
        backend: Optional[CPUBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        #: Unique, monotonically increasing id.  Used as a graph-node key.
        self.unique_id: int = _tensor_count

        if not isinstance(data, TensorData):
            raise TypeError(f"Tensor requires TensorData, got {type(data)!r}")
        self._tensor: TensorData = data
        #: Backend that executes this tensor's kernels (CPU by default).
        self.backend: CPUBackend = backend if backend is not None else default_backend
        #: ``None`` = constant, ``History()`` = user leaf, else built by a fn.
        self.history: Optional[History] = history
        #: Gradient accumulated by the last :meth:`backward` (a raw tensor).
        self.grad: Optional[Tensor] = None

    # -- Low-level factories ---------------------------------------------------
    @staticmethod
    def make(
        storage: Union[Sequence[float], np.ndarray],
        shape: Sequence[int],
        strides: Optional[Sequence[int]] = None,
        backend: Optional[CPUBackend] = None,
    ) -> "Tensor":
        """Build a tensor directly from flat storage + shape (no autograd)."""
        return Tensor(
            TensorData(storage, tuple(shape), tuple(strides) if strides else None),
            backend=backend,
        )

    def _new(self, data: TensorData) -> "Tensor":
        """Wrap raw ``data`` into a tensor sharing this tensor's backend."""
        return Tensor(data, backend=self.backend)

    def detach(self) -> "Tensor":
        """Return a tensor sharing storage but detached from the graph."""
        return Tensor(self._tensor, backend=self.backend)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        """Tensor dimensions."""
        return self._tensor.shape

    @property
    def strides(self) -> Tuple[int, ...]:
        """Memory layout strides."""
        return self._tensor.strides

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Number of dimensions."""
        return self._tensor.dims

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype of the storage (float32 by default)."""
        return self._tensor.dtype

    @property
    def device(self):
        """The device this tensor computes on (``cpu``/``numba`` → ``device('cpu')``,
        ``cuda`` → ``device('cuda')``)."""
        return self.backend.device

    def to(self, device=None):
        """
        Return a copy of this tensor moved to another backend.

        All Torchlight kernels operate on numpy-backed storage, so ``to`` is a
        cheap re-wrap of the underlying data with a new backend rather than
        a data copy.  Leaves keep ``requires_grad``; graph nodes are detached.
        """
        target = get_backend(device)
        if target is self.backend:
            return self
        out = Tensor(self._tensor, backend=target)
        if self.requires_grad and self.is_leaf():
            out.requires_grad_(True)
        return out

    @property
    def requires_grad(self) -> bool:
        """``True`` when this tensor participates in autodifferentiation."""
        return self.history is not None

    def is_leaf(self) -> bool:
        """``True`` for user-created leaves (requires grad, no parents)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """``True`` for values disconnected from the gradient graph."""
        return self.history is None

    @property
    def parents(self) -> Sequence["Tensor"]:
        """The tensors this one was computed from (requires non-constant)."""
        assert self.history is not None, "constants have no parents"
        return self.history.inputs

    def requires_grad_(self, flag: bool = True) -> "Tensor":
        """Toggle autograd participation in place; returns ``self``."""
        if flag and self.history is None:
            self.history = History()
        elif not flag:
            self.history = None
        return self

    # ------------------------------------------------------------------
    # Coercion helpers
    # ------------------------------------------------------------------
    def _ensure_tensor(self, x: TensorLike) -> "Tensor":
        """Coerce numbers / lists / raw data into a tensor on this backend."""
        if isinstance(x, Tensor):
            return x
        if isinstance(x, np.ndarray):
            return from_numpy(x, backend=self.backend)
        if isinstance(x, (list, tuple)):
            return tensor_nested(x, backend=self.backend)
        return Tensor.make([x], (1,), backend=self.backend)

    def _zeros_like(self) -> "Tensor":
        """Allocate a zero tensor with the (contiguous) shape of ``self``."""
        return Tensor(TensorData.zeros(self.shape), backend=self.backend)

    def zeros(self, shape: Optional[Sequence[int]] = None) -> "Tensor":
        """
        Allocate a zero tensor (shape defaults to ``self.shape``).

        Also callable class-style — ``Tensor.zeros((a, b))`` — which simply
        delegates to the module-level ``zeros`` factory.
        """
        if not isinstance(self, Tensor):
            shape, self = self, None
            return zeros(shape)  # module-level factory
        return Tensor(
            TensorData.zeros(tuple(shape) if shape is not None else self.shape),
            backend=self.backend,
        )

    def ones(self, shape: Optional[Sequence[int]] = None) -> "Tensor":
        """
        Allocate a ones tensor (shape defaults to ``self.shape``).

        Also callable class-style — ``Tensor.ones((a, b))`` — which delegates
        to the module-level ``ones`` factory.
        """
        if not isinstance(self, Tensor):
            shape, self = self, None
            return ones(shape)  # module-level factory
        return Tensor(
            TensorData.ones(tuple(shape) if shape is not None else self.shape),
            backend=self.backend,
        )

    def _raw_add(self, other: "Tensor") -> "Tensor":
        """Backend-level addition on raw data (no new graph edge)."""
        return self._new(self.backend.add(self._tensor, other._tensor))

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------
    def __add__(self, b: TensorLike) -> "Tensor":
        return Add.apply(self, self._ensure_tensor(b))

    def __radd__(self, b: TensorLike) -> "Tensor":
        return self + b

    def __sub__(self, b: TensorLike) -> "Tensor":
        return Add.apply(self, Neg.apply(self._ensure_tensor(b)))

    def __rsub__(self, b: TensorLike) -> "Tensor":
        return Add.apply(Neg.apply(self), self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> "Tensor":
        return Mul.apply(self, self._ensure_tensor(b))

    def __rmul__(self, b: TensorLike) -> "Tensor":
        return self * b

    def __truediv__(self, b: TensorLike) -> "Tensor":
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> "Tensor":
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __pow__(self, b: TensorLike) -> "Tensor":
        if isinstance(b, Tensor) and b.size == 1:
            return PowerScalar.apply(self, self._ensure_tensor(b.item()))
        return PowerScalar.apply(self, self._ensure_tensor(b))

    def __rpow__(self, b: TensorLike) -> "Tensor":
        return PowerScalar.apply(self._ensure_tensor(b), self)

    def __matmul__(self, b: TensorLike) -> "Tensor":
        return MatMul.apply(self, self._ensure_tensor(b))

    def __neg__(self) -> "Tensor":
        return Neg.apply(self)

    def __abs__(self) -> "Tensor":
        return Abs.apply(self)

    # In-place ops materialise a fresh tensor (never write into strided views).
    def __iadd__(self, b: TensorLike) -> "Tensor":
        return self + b

    def __isub__(self, b: TensorLike) -> "Tensor":
        return self - b

    def __imul__(self, b: TensorLike) -> "Tensor":
        return self * b

    # ------------------------------------------------------------------
    # Comparisons (return 0.0/1.0 float tensors)
    # ------------------------------------------------------------------
    def __lt__(self, b: TensorLike) -> "Tensor":
        return LT.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> "Tensor":
        return GT.apply(self, self._ensure_tensor(b))

    def __le__(self, b: TensorLike) -> "Tensor":
        return self._ensure_tensor(1) - LT.apply(self, self._ensure_tensor(b))

    def __ge__(self, b: TensorLike) -> "Tensor":
        return self._ensure_tensor(1) - LT.apply(self._ensure_tensor(b), self)

    def __eq__(self, other: object) -> "Tensor":  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(other))

    def __ne__(self, other: object) -> "Tensor":  # type: ignore[override]
        return self._ensure_tensor(1) - EQ.apply(self, self._ensure_tensor(other))

    def __bool__(self) -> bool:
        raise ValueError(
            "The truth value of a Tensor is ambiguous; use .item() or compare "
            "with an explicit shape."
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def __getitem__(self, key: Union[int, slice, Tuple[Union[int, slice, None, type(Ellipsis)], ...]]):
        """
        Index into this tensor.

        Two behaviours, matched to PyTorch conventions:

        * A complete tuple of integers (or a bare int on a 1-D tensor)
          returns the plain ``float`` at that position (legacy behaviour).
        * Any index containing ``slice``, ``Ellipsis`` or ``None`` returns a
          new :class:`Tensor`.  Integer/slice entries select along their
          source axis (``start:stop:step`` slicing with negative steps
          supported); ``None`` inserts a fresh size-1 axis; ``...`` pads with
          full slices; an integer entry on its own axis squeezes that axis
          out of the result.  The selection is differentiable (an
          :class:`~torchlight.autograd.indexing.IndexSelect` node), so
          gradients flow back to the selected positions.
        """
        key2 = (key,) if isinstance(key, (int, np.integer, slice, type(Ellipsis), type(None))) else tuple(key)

        # Expand ellipsis to the interior full slices it stands for.
        n_ell = key2.count(Ellipsis)
        if n_ell > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if n_ell == 1:
            at = key2.index(Ellipsis)
            missing = self.dims - (len(key2) - 1)
            if missing < 0:
                raise IndexError(f"too many indices for tensor of dimension {self.dims}")
            key2 = key2[:at] + (slice(None),) * missing + key2[at + 1 :]

        # Nothing may consume more axes than the tensor has (``None`` is free).
        if sum(1 for k in key2 if not isinstance(k, type(None))) > self.dims:
            raise IndexError(f"too many indices for tensor of dimension {self.dims}")

        # Legacy fast path: a full tuple of integers -> scalar float.
        if all(isinstance(k, (int, np.integer)) for k in key2) and len(key2) == self.dims:
            normalized = []
            for k, n in zip(key2, self.shape):
                i = int(k)
                if i < 0:
                    i += n
                if not 0 <= i < n:
                    raise IndexError(f"index {k} is out of bounds for dimension with size {n}")
                normalized.append(i)
            return self._tensor.get(tuple(normalized))

        return self._basic_slice(key2)

    def _basic_slice(self, key2: Tuple) -> "Tensor":
        """
        Evaluate a per-axis slice/int/None selection as a differentiable
        :class:`Tensor` (internal helper for ``__getitem__``).

        Positions are relative to the running result: every processed entry
        contributes exactly one axis (slices keep theirs, ``None`` inserts a
        size-1 axis, ints leave a size-1 axis that is squeezed at the end),
        so the next operation always applies at ``di`` = len(processed prefix).
        """
        out = self
        di = 0  # position in the current result == len(processed prefix)
        src = 0  # which source axis the next slice/int consumes
        for k in key2:
            if k is None:
                out = out.unsqueeze(di)
                di += 1
                continue
            n = self.shape[src]
            if isinstance(k, slice):
                start, stop, step = k.indices(n)
                idx = np.arange(start, stop, step)
            else:
                i = int(k)
                if i < 0:
                    i += n
                if not 0 <= i < n:
                    raise IndexError(f"index {k} is out of bounds for dimension {src} with size {n}")
                idx = np.array([i], dtype=np.int64)
            idx_t = Tensor.make(idx.tolist(), idx.shape, backend=out.backend)
            out = IndexSelect.apply(out, idx_t, out._ensure_tensor(di))
            src += 1
            di += 1

        # Squeeze out the size-1 axes left by integer entries.
        target = []
        si = 0
        for k in key2:
            if k is None:
                target.append(1)
            elif isinstance(k, (int, np.integer)):
                si += 1
            else:
                start, stop, step = k.indices(self.shape[si])
                si += 1
                target.append(len(range(start, stop, step)))
        target.extend(self.shape[si:])
        if out.shape != tuple(target):
            out = out.reshape(*target)
        return out

    def index_select(self, dim: int, index: TensorLike) -> "Tensor":
        """
        Select along ``dim`` at the positions given by the 1-D ``index``
        (differentiable; gradients accumulate when indices repeat).
        """
        index_t = self._ensure_tensor(index).contiguous()
        if index_t.dims != 1:
            raise ValueError(f"index_select index must be 1-D, got shape {index_t.shape}")
        return IndexSelect.apply(self, index_t, self._ensure_tensor(dim))

    def __setitem__(self, key: Union[int, Tuple[int, ...]], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else tuple(key)
        self._tensor.set(key2, float(val))

    @property
    def data_(self) -> np.ndarray:
        """
        Direct mutable view of the underlying storage (for initializers).

        Note: writing into a non-contiguous view would corrupt memory layout,
        so this returns a *contiguous* storage segment.  Parameters are always
        contiguous, which is what ``nn.init`` uses.
        """
        return self._tensor.storage

    # -- In-place initializers (used by torchlight.nn.init) ---------------------
    def fill_(self, value: float) -> "Tensor":
        """Fill all elements with ``value`` in place."""
        self._tensor.storage[:] = value
        return self

    def zeros_(self) -> "Tensor":
        """Zero-fill in place."""
        self._tensor.storage[:] = 0.0
        return self

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> "Tensor":
        """Fill in place from a uniform distribution."""
        self._tensor.storage[:] = np.random.uniform(low, high, self.size).astype(np.float32)
        return self

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> "Tensor":
        """Fill in place from a normal distribution."""
        self._tensor.storage[:] = np.random.normal(mean, std, self.size).astype(np.float32)
        return self

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def item(self) -> float:
        """Return the single value of a 1-element tensor."""
        assert self.size == 1, f"item() only valid for 1-element tensors, got {self.shape}"
        return float(self._tensor.storage[0])

    def to_numpy(self) -> np.ndarray:
        """Return a dense (contiguous) numpy copy of the data."""
        return self._tensor.to_numpy()

    def clone(self) -> "Tensor":
        """Return a detached copy with the same data (no shared memory)."""
        return self._new(self._tensor.contiguous())

    # ------------------------------------------------------------------
    # Elementwise math
    # ------------------------------------------------------------------
    def exp(self) -> "Tensor":
        return Exp.apply(self)

    def log(self) -> "Tensor":
        return Log.apply(self)

    def sigmoid(self) -> "Tensor":
        return Sigmoid.apply(self)

    def tanh(self) -> "Tensor":
        return Tanh.apply(self)

    def abs(self) -> "Tensor":
        return Abs.apply(self)

    def relu(self) -> "Tensor":
        return ReLU.apply(self)

    def sqrt(self) -> "Tensor":
        return Sqrt.apply(self)

    def clamp(self, lo: TensorLike = 0.0, hi: TensorLike = 1.0) -> "Tensor":
        r"""
        Clamp all values to ``[lo, hi]`` (differentiable).

        Composed from ReLU as ``min(max(x, lo), hi)``, so gradients flow
        through the already-implemented ops.
        """
        lo_t, hi_t = self._ensure_tensor(lo), self._ensure_tensor(hi)
        lowered = lo_t + (self - lo_t).relu()
        return hi_t - (hi_t - lowered).relu()

    def rand_like(self) -> "Tensor":
        """Uniform random tensor in ``[0, 1)`` with the same shape."""
        data = np.random.rand(*self.shape).astype(np.float32)
        return Tensor(TensorData(data, tuple(self.shape)), backend=self.backend)

    def attn_softmax(self, mask: TensorLike) -> "Tensor":
        r"""
        Row-wise softmax of ``self + mask`` over the last axis (fused kernel).

        Mirrors the fused attention-softmax kernels (``exp(z - max)`` with a
        ``1e-8`` guard on the sum).  ``mask`` is numpy-broadcastable to
        ``self`` -- e.g. ``(S,)`` or ``(B,1,1,S)`` for ``(B,H,T,S)`` scores --
        and is treated as a constant (no gradient flows to it).
        """
        mask_t = self._ensure_tensor(mask)
        return AttnSoftmax.apply(self, mask_t)

    def layernorm(self, gamma: TensorLike, beta: TensorLike) -> "Tensor":
        r"""
        Layer normalisation over the last axis:

        .. math:: y = \frac{x - \mathrm{mean}(x)}{\sqrt{\mathrm{var}(x) + 10^{-5}}}
                        \odot \gamma + \beta

        ``gamma``/``beta`` are per-feature gain/bias with shape equal to the
        last axis; both receive gradients.
        """
        gamma_t = self._ensure_tensor(gamma).contiguous()
        beta_t = self._ensure_tensor(beta).contiguous()
        if gamma_t.dims != 1 or gamma_t.shape[0] != self.shape[-1]:
            raise ValueError(
                f"layernorm gamma/beta must be shape ({self.shape[-1]},), got {gamma_t.shape}"
            )
        return LayerNorm.apply(self, gamma_t, beta_t)

    def _gather_index(self, index: TensorLike, dim: int) -> "Tensor":
        r"""
        Gather elements along ``dim`` at the positions given by ``index``
        (used by ``cross_entropy``/``nll_loss`` with index targets).

        ``index`` must have one fewer dimension than ``self`` (the gathered
        dim is implicit); a singleton dim is inserted so numpy's
        ``take_along_axis`` works, and the result has the shape of ``index``.
        """
        index_t = self._ensure_tensor(index).contiguous()
        if index_t.dims == self.dims - 1:
            shape = list(index_t.shape)
            shape.insert(dim, 1)
            index_t = index_t.contiguous().view(*shape)
        return Gather.apply(self, index_t, self._ensure_tensor(dim))

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------
    def sum(self, dim: Optional[int] = None) -> "Tensor":
        """
        Sum along ``dim`` (kept as size 1) or over everything if ``None``.

        Returns a tensor with shape ``(...)`` minus ``dim`` (reduced to 1),
        or a ``(1,)`` scalar when ``dim is None``.
        """
        if dim is None:
            # Flatten then reduce dimension 0 -> shape (1,).
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> "Tensor":
        """Arithmetic mean along ``dim`` or over everything."""
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        return self.sum() / self.size

    def var(self, dim: Optional[int] = None) -> "Tensor":
        """Population variance along ``dim`` or over everything."""
        if dim is not None:
            count = self.shape[dim]
            mean = self.sum(dim) / count
            return ((self - mean) ** 2).sum(dim) / count
        count = self.size
        mean = self.sum() / count
        return ((self - mean) ** 2).sum() / count

    def std(self, dim: Optional[int] = None) -> "Tensor":
        """Standard deviation (sqrt of population variance)."""
        return self.var(dim).sqrt()

    def max(self, dim: Optional[int] = None) -> "Tensor":
        """Maximum along ``dim`` (kept as size 1)."""
        if dim is None:
            return Max.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        return Max.apply(self, self._ensure_tensor(dim))

    def min(self, dim: Optional[int] = None) -> "Tensor":
        """Minimum along ``dim`` (kept as size 1)."""
        if dim is None:
            return Min.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        return Min.apply(self, self._ensure_tensor(dim))

    def all(self, dim: Optional[int] = None) -> "Tensor":
        """Product-reduction (all-nonzero as floats) along ``dim``."""
        if dim is None:
            return All.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        return All.apply(self, self._ensure_tensor(dim))

    def any(self, dim: Optional[int] = None) -> "Tensor":
        """``1.0`` if any element is non-zero (derived from ``max``)."""
        return self.max(dim) > 0 if dim is not None else self.max() > 0

    def is_close(self, y: TensorLike) -> "Tensor":
        """Elementwise closeness (``|a - b| < 1e-2``)."""
        return IsClose.apply(self, self._ensure_tensor(y))

    # ------------------------------------------------------------------
    # Layout ops
    # ------------------------------------------------------------------
    def view(self, *shape: int) -> "Tensor":
        """Reshape to ``shape`` (requires contiguous data)."""
        shape_t = Tensor.make(list(shape), (len(shape),), backend=self.backend)
        return View.apply(self, shape_t)

    def reshape(self, *shape: int) -> "Tensor":
        """Reshape to ``shape``, materialising contiguity if needed."""
        return self.contiguous().view(*shape)

    def unsqueeze(self, dim: int) -> "Tensor":
        """Insert a size-1 axis at position ``dim`` (may be negative)."""
        if dim < 0:
            dim += self.dims + 1
        if not 0 <= dim <= self.dims:
            raise IndexError(f"dim {dim} out of range [{-self.dims - 1}, {self.dims}] for {self.shape}")
        shape = list(self.shape)
        shape.insert(dim, 1)
        return self.reshape(*shape)

    def permute(self, *order: int) -> "Tensor":
        """Reorder dimensions (a strided view, no copy)."""
        order_t = Tensor.make(list(order), (len(order),), backend=self.backend)
        return Permute.apply(self, order_t)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """Swap two dimensions."""
        order = list(range(self.dims))
        order[dim0], order[dim1] = order[dim1], order[dim0]
        return self.permute(*order)

    def contiguous(self) -> "Tensor":
        """Return a dense copy (identity if already contiguous)."""
        return Contiguous.apply(self)

    def flatten(self) -> "Tensor":
        """Flatten to a 1-D tensor of ``size`` elements."""
        return self.contiguous().view(self.size)

    # ------------------------------------------------------------------
    # Gradient interface (mirrors torch API)
    # ------------------------------------------------------------------
    def backward(self, grad_output: Optional["Tensor"] = None) -> None:
        """
        Run reverse-mode autodiff from this tensor's graph root.

        Args:
            grad_output: gradient to seed (must be ``value``-like for a
                scalar root; defaults to ones for ``shape == (1,)``).
        """
        if grad_output is None:
            assert self.shape == (1,), (
                f"backward() requires a scalar root or an explicit grad_output, "
                f"got shape {self.shape}"
            )
            grad_output = self._zeros_like() + 1.0
        backpropagate(self, grad_output)

    def zero_grad_(self) -> None:
        """Reset the accumulated gradient."""
        self.grad = None

    # -- Internal methods consumed by the autodiff engine -----------------------
    def accumulate_derivative(self, x: Any) -> None:
        """Add ``x`` to this leaf's accumulated gradient."""
        assert self.is_leaf(), "Only leaf variables can accumulate derivatives."
        x = x.detach()
        if self.grad is None:
            self.grad = self._zeros_like()
        self.grad = self.grad._raw_add(x)

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Tensor", "Tensor"]]:
        """Push ``d_output`` through this node, yielding (parent, d_parent)."""
        h = self.history
        assert h is not None, "constants have no chain rule"
        assert h.last_fn is not None and h.ctx is not None
        grads = h.last_fn._backward(h.ctx, d_output)  # type: ignore[attr-defined]
        assert len(grads) == len(h.inputs), f"Bug in Function {h.last_fn}"
        return [
            (inp, inp._expand_grad(self._ensure_tensor(d_in), inp.shape))
            for inp, d_in in zip(h.inputs, grads)
        ]

    def _expand_grad(self, g: "Tensor", target_shape: Tuple[int, ...]) -> "Tensor":
        """
        Bring a backwards gradient ``g`` into ``target_shape``.

        Because ``forward`` may have broadcast, the raw gradient can be
        smaller than the input it belongs to.  The fix is: broadcast ``g`` up,
        then sum over the dims where ``target_shape`` is 1 but the broadcast
        shape is not -- exactly what backprop over broadcasting requires.
        """
        if g.shape == target_shape:
            return g

        common = shape_broadcast(g.shape, target_shape)
        buf = Tensor(g._tensor.broadcast_to(common), backend=g.backend)
        orig_shape = [1] * (len(common) - len(target_shape)) + list(target_shape)

        cur = buf
        for d, o in enumerate(orig_shape):
            if o == 1 and common[d] != 1:
                cur = Sum.apply(cur, self._ensure_tensor(d))
        if cur.shape != target_shape:
            cur = cur.contiguous().view(*target_shape)
        return cur

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        arr = self._tensor.to_numpy()
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor({arr.tolist()}{grad})"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Construction helpers (mirror torch.tensor / zeros / ones ...)
# ---------------------------------------------------------------------------
def _resolve_backend(backend=None, device=None):
    """Combine the legacy ``backend=`` kwarg with the newer ``device=`` kwarg.

    ``backend`` wins if both are given; ``device`` may be a string, a
    :class:`Device`, or a backend instance (resolved via
    :func:`~torchlight.backends.get_backend`).
    """
    if backend is not None:
        return backend
    if device is not None:
        return get_backend(device)
    return None


def _infer_shape(ls: Any) -> List[int]:
    """Recursively infer the shape of a nested list/tuple."""
    if isinstance(ls, (list, tuple)):
        return [len(ls)] + _infer_shape(ls[0]) if len(ls) else [0]
    return []


def _flatten(ls: Any) -> List[float]:
    """Recursively flatten a nested list/tuple into 1-D."""
    if isinstance(ls, (list, tuple)):
        return [y for x in ls for y in _flatten(x)]
    return [float(ls)]


def from_numpy(
    arr: np.ndarray,
    backend: Optional[CPUBackend] = None,
    requires_grad: bool = False,
    device=None,
) -> Tensor:
    """
    Create a tensor from a numpy array.  Accepts any dtype (cast to float32).
    """
    backend = _resolve_backend(backend, device)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    out = Tensor(TensorData(arr.reshape(-1), tuple(arr.shape)), backend=backend)
    return out.requires_grad_(requires_grad)


def tensor(
    ls: Union[float, Sequence[Any], np.ndarray],
    backend: Optional[CPUBackend] = None,
    requires_grad: bool = False,
    device=None,
) -> Tensor:
    """
    Create a tensor from a python number, list, tuple, or numpy array.

        >>> x = tensor([[1, 2], [3, 4]])
        >>> x.shape
        (2, 2)
    """
    backend = _resolve_backend(backend, device)
    if isinstance(ls, np.ndarray):
        return from_numpy(ls, backend=backend, requires_grad=requires_grad)
    shape = tuple(_infer_shape(ls))
    flat = np.asarray(_flatten(ls), dtype=np.float32)
    out = Tensor(TensorData(flat, shape), backend=backend)
    return out.requires_grad_(requires_grad)


#: Alias matching numpy terminology.
tensor_nested = tensor


def _normalize_shape(shape: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    """Accept ``3`` or ``(3,)`` / ``[3]`` alike; return a tuple of ints."""
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(s) for s in shape)


def zeros(
    shape: Union[int, Sequence[int]], backend: Optional[CPUBackend] = None, requires_grad: bool = False, device=None
) -> Tensor:
    """All-zero tensor of ``shape``."""
    backend = _resolve_backend(backend, device)
    t = Tensor(TensorData.zeros(_normalize_shape(shape)), backend=backend)
    return t.requires_grad_(requires_grad)


def ones(
    shape: Union[int, Sequence[int]], backend: Optional[CPUBackend] = None, requires_grad: bool = False, device=None
) -> Tensor:
    """All-once tensor of ``shape``."""
    backend = _resolve_backend(backend, device)
    t = Tensor(TensorData.ones(_normalize_shape(shape)), backend=backend)
    return t.requires_grad_(requires_grad)


def empty(
    shape: Union[int, Sequence[int]], backend: Optional[CPUBackend] = None, requires_grad: bool = False, device=None
) -> Tensor:
    """Uninitialised tensor of ``shape``."""
    backend = _resolve_backend(backend, device)
    t = Tensor(TensorData.empty(_normalize_shape(shape)), backend=backend)
    return t.requires_grad_(requires_grad)


def full(
    shape: Union[int, Sequence[int]],
    value: float,
    backend: Optional[CPUBackend] = None,
    requires_grad: bool = False,
    device=None,
) -> Tensor:
    """Tensor of ``shape`` filled with ``value``."""
    backend = _resolve_backend(backend, device)
    shape = _normalize_shape(shape)
    storage = np.full(prod(shape), float(value), dtype=np.float32)
    t = Tensor(TensorData(storage, shape), backend=backend)
    return t.requires_grad_(requires_grad)


def rand(
    shape: Union[int, Sequence[int]], backend: Optional[CPUBackend] = None, requires_grad: bool = False, device=None
) -> Tensor:
    """Uniform random tensor in ``[0, 1)``."""
    backend = _resolve_backend(backend, device)
    shape = _normalize_shape(shape)
    t = Tensor(
        TensorData(np.random.rand(*shape).astype(np.float32), shape),
        backend=backend,
    )
    return t.requires_grad_(requires_grad)


def randn(
    shape: Union[int, Sequence[int]], backend: Optional[CPUBackend] = None, requires_grad: bool = False, device=None
) -> Tensor:
    """Standard-normal random tensor."""
    backend = _resolve_backend(backend, device)
    shape = _normalize_shape(shape)
    t = Tensor(
        TensorData(np.random.randn(*shape).astype(np.float32), shape),
        backend=backend,
    )
    return t.requires_grad_(requires_grad)


def arange(
    start: float, stop: Optional[float] = None, step: float = 1.0, requires_grad: bool = False, device=None
) -> Tensor:
    """Values evenly spaced ``[start, stop)`` (numpy-compatible signature)."""
    if stop is None:
        start, stop = 0.0, start
    values = np.arange(start, stop, step, dtype=np.float32)
    t = Tensor(
        TensorData(values, values.shape),
        backend=_resolve_backend(None, device),
    )
    return t.requires_grad_(requires_grad)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Concatenate ``tensors`` along ``dim`` (differentiable)."""
    ts = list(tensors)
    if not ts:
        raise ValueError("cat() requires at least one tensor")
    dim_t = ts[0]._ensure_tensor(dim)
    return Cat.apply(*ts, dim_t)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along a *new* axis ``dim`` (differentiable)."""
    ts = list(tensors)
    if not ts:
        raise ValueError("stack() requires at least one tensor")
    shape = list(ts[0].shape)
    dim = dim if dim >= 0 else dim + len(shape) + 1
    reshaped = []
    for t in ts:
        s = list(t.shape)
        s.insert(dim, 1)
        reshaped.append(t.reshape(*s))
    return cat(reshaped, dim=dim)


def split(tensor: Tensor, split_size: int, dim: int = 0) -> List[Tensor]:
    """Split ``tensor`` into chunks of at most ``split_size`` along ``dim``."""
    n = tensor.shape[dim]
    out: List[Tensor] = []
    for i in range(0, n, split_size):
        idx = arange(i, min(i + split_size, n), requires_grad=False, device=tensor.device)
        out.append(tensor.index_select(dim, idx))
    return out


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> List[Tensor]:
    """Split ``tensor`` into ``chunks`` roughly equal parts along ``dim``."""
    n = tensor.shape[dim]
    sizes = [n // chunks] * chunks
    for i in range(n % chunks):
        sizes[i] += 1
    out: List[Tensor] = []
    off = 0
    for s in sizes:
        idx = arange(off, off + s, requires_grad=False, device=tensor.device)
        out.append(tensor.index_select(dim, idx))
        off += s
    return out


__all__ = [
    "Tensor", "tensor", "from_numpy", "zeros", "ones", "empty", "full",
    "rand", "randn", "arange", "cat", "stack", "split", "chunk",
]