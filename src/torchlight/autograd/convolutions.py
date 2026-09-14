"""
Differentiable convolution ops (direct numpy implementation).

``Conv1d`` / ``Conv2d`` are :class:`Function` subclasses that implement a
*direct* convolution and its backward, supporting arbitrary ``stride`` and
``padding``.

Forward sliding-window view + einsum:

.. math::
    out[n,o,l] = sum_{c,k} x[n,c,l*s + k - p] * w[o,c,k]

Backward reuses the same windows: the weight gradient is the correlation of
the (padded) input windows with the output gradient, and the input gradient
scatters each window's contribution back (at ``l*s + k``) before unpadding --
exactly the transpose of forward.

These ops run on raw :class:`~torchlight.core.tensor_data.TensorData` (like
``Gather``), so they are CPU-only for now; the functional wrappers in
``nn.functional`` expose them with autograd wired through ``apply``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..core.tensor_data import TensorData
from .autodiff import Context
from .functions import _as_int, Function


class _ConvNd(Function):
    """Shared plumbing for the 1D / 2D convolution Functions."""

    @staticmethod
    def _out_len(inp: int, kernel: int, stride: int, padding: int) -> int:
        """Output length for a strided, padded 1D length."""
        return (inp + 2 * padding - kernel) // stride + 1

    @staticmethod
    def _result(t: Any, arr: np.ndarray) -> Any:
        """Rebuild a graph node carrying ``arr`` on tensor ``t``'s backend."""
        return t._new(TensorData(np.ascontiguousarray(arr, dtype=np.float32), tuple(arr.shape)))


class Conv1d(_ConvNd):
    r"""
    Cross-correlation of a ``(N, C_in, L)`` input with a ``(C_out, C_in, K)``
    weight, producing ``(N, C_out, L_out)`` where
    :math:`L_{out} = (L + 2p - K)/s + 1`.

    ``stride`` / ``padding`` are passed as 1-element constant tensors (as the
    framework metadata convention).  The output is the *correlation* (no
    kernel flip), matching PyTorch's ``conv1d``.
    """

    @staticmethod
    def _windows(x: np.ndarray, kernel: int, stride: int, padding: int) -> Tuple[np.ndarray, int]:
        """Padded input reshaped into ``(N, C_in, L_out, K)`` sliding windows."""
        xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
        win = sliding_window_view(xp, kernel, axis=-1)
        return win[:, :, ::stride, :], xp.shape[-1]

    @staticmethod
    def _forward_impl(x: np.ndarray, w: np.ndarray, stride: int, padding: int) -> np.ndarray:
        win = Conv1d._windows(x, w.shape[2], stride, padding)[0]
        return np.einsum("nclk,ock->nol", win, w)

    @staticmethod
    def forward(ctx: Context, input: Any, weight: Any, stride: Any, padding: Any) -> Any:
        x, w = input._tensor.to_numpy(), weight._tensor.to_numpy()
        s, p = _as_int(stride), _as_int(padding)
        out = Conv1d._forward_impl(x, w, s, p)
        ctx.save_for_backward(input, weight, s, p)
        return Conv1d._result(input, out)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any, Any]:
        input, weight, stride, padding = ctx.saved_tensors
        x, w = input._tensor.to_numpy(), weight._tensor.to_numpy()
        dout = grad_output._tensor.to_numpy()

        n, c, l = x.shape
        k = w.shape[2]
        lout = dout.shape[2]

        win, xp_len = Conv1d._windows(x, k, stride, padding)
        # Weights: correlate the input windows with the output gradient.
        dw = np.einsum("nclk,nol->ock", win, dout)
        # Input: scatter each window's contribution back to l*s + k.
        # dxp[:, :, kk::stride] indexes positions kk + l*stride; only the first
        # ``lout`` of them are real output slots (the rest lie past the input).
        dwin = np.einsum("nol,ock->nclk", dout, w)
        dxp = np.zeros((n, c, xp_len), dtype=np.float32)
        for kk in range(k):
            dxp[:, :, kk::stride][:, :, :lout] += dwin[:, :, :, kk]
        dx = dxp[:, :, padding : padding + l]

        return (
            Conv1d._result(input, dx),
            Conv1d._result(weight, dw),
            0.0,
            0.0,
        )


class Conv2d(_ConvNd):
    r"""
    Cross-correlation of a ``(N, C_in, H, W)`` input with a
    ``(C_out, C_in, KH, KW)`` weight, producing ``(N, C_out, H_out, W_out)``.
    ``stride`` / ``padding`` are each passed as a 2-element constant tensor.
    """

    @staticmethod
    def _windows(
        x: np.ndarray,
        kh: int,
        kw: int,
        sh: int,
        sw: int,
        ph: int,
        pw: int,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        win = sliding_window_view(xp, (kh, kw), axis=(-2, -1))
        return win[:, :, ::sh, ::sw, :, :], xp.shape[-2:]

    @staticmethod
    def _forward_impl(
        x: np.ndarray,
        w: np.ndarray,
        sh: int,
        sw: int,
        ph: int,
        pw: int,
    ) -> np.ndarray:
        win = Conv2d._windows(x, w.shape[2], w.shape[3], sh, sw, ph, pw)[0]
        return np.einsum("nchwab,ocab->nohw", win, w)

    @staticmethod
    def forward(ctx: Context, input: Any, weight: Any, stride: Any, padding: Any) -> Any:
        x, w = input._tensor.to_numpy(), weight._tensor.to_numpy()
        sh, sw = (int(stride[0]), int(stride[1]))
        ph, pw = (int(padding[0]), int(padding[1]))
        out = Conv2d._forward_impl(x, w, sh, sw, ph, pw)
        ctx.save_for_backward(input, weight, sh, sw, ph, pw)
        return Conv2d._result(input, out)

    @staticmethod
    def backward(ctx: Context, grad_output: Any) -> Tuple[Any, Any, Any]:
        input, weight, sh, sw, ph, pw = ctx.saved_tensors
        x, w = input._tensor.to_numpy(), weight._tensor.to_numpy()
        dout = grad_output._tensor.to_numpy()

        n, c, h, ww = x.shape
        kh, kw = w.shape[2], w.shape[3]
        hout, wout = dout.shape[2], dout.shape[3]

        win, (hp, wp) = Conv2d._windows(x, kh, kw, sh, sw, ph, pw)
        dw = np.einsum("nchwab,nohw->ocab", win, dout)
        dwin = np.einsum("nohw,ocab->nchwab", dout, w)
        dxp = np.zeros((n, c, hp, wp), dtype=np.float32)
        for khh in range(kh):
            for kww in range(kw):
                dxp[:, :, khh::sh, kww::sw][:, :, :hout, :wout] += dwin[:, :, :, :, khh, kww]
        dx = dxp[:, :, ph : ph + h, pw : pw + ww]

        return (
            Conv2d._result(input, dx),
            Conv2d._result(weight, dw),
            0.0,
            0.0,
        )


__all__ = ["Conv1d", "Conv2d"]