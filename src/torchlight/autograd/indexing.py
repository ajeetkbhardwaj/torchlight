"""
Indexing operations (``Function`` subclasses).

The whole family of "pick rows by indices" behaviour -- ``index_select``,
``split`` / ``chunk``, basic ``x[i : j : k]`` slicing and embedding lookups --
reduces to one operation, :class:`IndexSelect`, plus :class:`Cat` for joining
tensors back together.

Both operators run on the *host* numpy copy of the data (the shared
:class:`~torchlight.core.tensor_data.TensorData` layout that every backend --
CPU, numba, CUDA -- operates on), so they behave identically on every device.
The forward pass is ``np.take`` / ``np.concatenate``; the backward pass
scatters gradients back with ``np.add.at`` (which correctly accumulates
buffered updates when the same index repeats) and ``np.split``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..core.tensor_data import TensorData
from .autodiff import Context
from .functions import Function, _as_int


class IndexSelect(Function):
    r"""
    Select a 1-D index along one dimension (``torch.index_select`` semantics).

    ``forward`` is ``np.take(arr, index, axis=dim)``; ``backward`` scatters
    the gradient back with ``np.add.at``, so the *sum* of repeated indices
    doubles up exactly like PyTorch.

    ``index`` is a 1-D constant tensor of integer positions (accepted as any
    int-valued tensor; the values are read via ``to_numpy()``).  ``dim`` is
    passed as a 1-element constant tensor, mirroring the framework's
    convention for reduction/layout ops.
    """

    @staticmethod
    def forward(ctx: Context, src: Any, index: Any, dim: Any) -> Any:
        dim_i = _as_int(dim)
        idx = index._tensor.to_numpy().astype(np.int64)
        ctx.save_for_backward(src, idx, dim_i)
        arr = src._tensor.to_numpy()
        out = np.take(arr, idx, axis=dim_i)
        return src._new(TensorData(np.ascontiguousarray(out), tuple(out.shape)))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any, Any]:
        src, idx, dim_i = ctx.saved_tensors
        grad = np.zeros(src.shape, dtype=np.float32)
        dout = np.ascontiguousarray(grad_output._tensor.to_numpy())
        if dim_i != 0:
            # Move the indexed axis to the front so ``np.add.at(grad, idx, dout)``
            # broadcasts over the remaining leading axes with the 1-D index.
            grad = np.moveaxis(grad, dim_i, 0)
            dout = np.moveaxis(dout, dim_i, 0)
        np.add.at(grad, idx, dout)
        if dim_i != 0:
            grad = np.moveaxis(grad, 0, dim_i)
        return (
            src._new(TensorData(np.ascontiguousarray(grad), tuple(src.shape))),
            0.0,
            0.0,
        )


class Cat(Function):
    r"""
    Concatenate a sequence of tensors along one dimension.

    A variadic form of the usual ``torch.cat``: ``forward`` is
    ``np.concatenate`` and ``backward`` is ``np.split`` along ``dim`` with the
    per-input boundary lengths recorded during the forward pass.  All inputs
    must share every dimension except ``dim`` (exactly like PyTorch).

    Callers invoke it as ``Cat.apply(*tensors, dim_tensor, ...)`` -- the
    trailing argument is the (constant) 1-element ``dim`` tensor.
    """

    @staticmethod
    def forward(ctx: Context, *vals: Any) -> Any:
        dim = _as_int(vals[-1])
        tensors = tuple(vals[:-1])
        if len(tensors) == 0:
            raise ValueError("cat() requires at least one tensor")
        shapes = tuple(t.shape[dim] for t in tensors)
        out = np.concatenate([t._tensor.to_numpy() for t in tensors], axis=dim)
        ctx.save_for_backward(tensors, shapes, dim)
        return tensors[0]._new(TensorData(np.ascontiguousarray(out), tuple(out.shape)))

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, ...]:
        tensors, shapes, dim = ctx.saved_tensors
        dout = grad_output._tensor.to_numpy()
        pieces = np.split(dout, np.cumsum(shapes)[:-1], axis=dim)
        return tuple(
            t._new(TensorData(np.ascontiguousarray(p), tuple(t.shape)))
            for t, p in zip(tensors, pieces)
        ) + (0.0,)


__all__ = ["IndexSelect", "Cat"]