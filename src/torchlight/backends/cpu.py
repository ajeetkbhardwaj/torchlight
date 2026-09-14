"""
CPU backend.

The backend is the one place in Torchlight that talks to the hardware (numpy
on CPU).  It exposes exactly six higher-order building blocks:

* ``map(fn)``    -- unary elementwise op over a :class:`TensorData`
* ``zip(fn)``    -- binary elementwise op (with broadcasting)
* ``reduce(fn, start)`` -- reduce one dimension
* ``matmul(a, b)``      -- (batched) matrix product
* ``allclose(a, b)``    -- numeric closeness helper

The autograd functions in :mod:`torchlight.autograd.functions` are written
against these six primitives, so swapping in a different backend later means
replacing *this file only*.

Implementation note:
    Every kernel is vectorised numpy.  Non-contiguous tensors (from ``permute``
    or ``broadcast_to``) are materialised as *strided views* via
    :meth:`TensorData.numpy_view` and fed straight into numpy ufuncs -- no
    python-level loops over elements are ever needed.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ..core.device import Device, cpu as cpu_device
from ..core.tensor_data import TensorData, shape_broadcast

# ---------------------------------------------------------------------------
# Vectorised (numpy) implementations of the scalar math ops.
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """Numerically stable sigmoid.

    ``np.where`` still evaluates *both* sides, so ``exp`` overflows for large
    ``|x|``; the ``errstate`` guard silences the resulting warnings while the
    branch selection stays numerically correct.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


_SIGMOID = _sigmoid  # noqa: E305
_LOG = lambda x: np.log(x + 1e-6)  # noqa: E731  (shift keeps log(0) finite)
_RELU = lambda x: np.maximum(x, 0.0)  # noqa: E731
_LEAKY = lambda alpha: (lambda x: np.where(x > 0, x, alpha * x))  # noqa: E731


def _make_leaky(alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return an array function computing :math:`leaky_relu(\\cdot, alpha)`."""
    return lambda x: np.where(x > 0, x, alpha * x)


def _sqrt_back(x, d):
    r"""Vectorized :math:`d / (2\sqrt{x})`, zero where :math:`x \le 0`."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = d / (2.0 * np.sqrt(x))
    return np.where(x > 0, out, 0.0)


#: Registry of available array map kernels: name -> vectorised function.
ARRAY_MAP: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "id": lambda x: x.copy(),
    "neg": np.negative,
    "inv": np.reciprocal,
    "sigmoid": _SIGMOID,
    "tanh": np.tanh,
    "relu": _RELU,
    "log": _LOG,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
}

#: Registry of available array zip kernels: name -> binary vectorised function.
ARRAY_ZIP: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "lt": np.less,
    "gt": np.greater,
    "eq": np.equal,
    "pow": np.power,
    "is_close": lambda a, b: np.abs(a - b) < 1e-2,
    "relu_back": lambda x, d: np.where(x > 0, d, 0.0),
    "log_back": lambda x, d: d / (x + 1e-6),
    "inv_back": lambda x, d: -d / (x * x),
    "sigmoid_back": lambda s, d: s * (1.0 - s) * d,
    "sqrt_back": _sqrt_back,
    "abs_back": lambda x, d: np.sign(x) * d,
}

#: Reduction kernels: (ufunc, start value).  ``np.add.reduce`` etc.
ARRAY_REDUCE: Dict[str, tuple] = {
    "sum": (np.add, 0.0),
    "prod": (np.multiply, 1.0),
    "max": (np.maximum, -np.inf),
    "min": (np.minimum, np.inf),
}


class CPUBackend:
    """
    Concrete numpy-on-CPU implementation of the six tensor primitives.

    An instance of this class is attached to every :class:`Tensor` as
    ``tensor.backend`` (and ``tensor.f``).  Ops always take/return
    :class:`TensorData` objects.

    Bound op names (used by the autograd layer)::

        backend.neg_map, backend.inv_map, backend.sigmoid_map, backend.tanh_map,
        backend.relu_map, backend.log_map, backend.exp_map, backend.sqrt_map,
        backend.abs_map                     # maps
        backend.add, backend.mul, backend.lt, backend.eq, backend.pow,
        backend.relu_back, backend.log_back, backend.inv_back,
        backend.sigmoid_back                # zips
        backend.add_reduce, backend.mul_reduce, backend.max_reduce,
        backend.min_reduce                  # reduces
        backend.matmul, backend.allclose    # multi-input
    """

    #: Hardware this backend talks to.
    device: Device = cpu_device
    is_cpu: bool = True
    is_cuda: bool = False

    def __init__(self) -> None:
        # --- Bind the higher-order primitives to concrete numpy kernels. ---
        self.neg_map = self.map(ARRAY_MAP["neg"])
        self.inv_map = self.map(ARRAY_MAP["inv"])
        self.sigmoid_map = self.map(ARRAY_MAP["sigmoid"])
        self.tanh_map = self.map(ARRAY_MAP["tanh"])
        self.relu_map = self.map(ARRAY_MAP["relu"])
        self.log_map = self.map(ARRAY_MAP["log"])
        self.exp_map = self.map(ARRAY_MAP["exp"])
        self.sqrt_map = self.map(ARRAY_MAP["sqrt"])
        self.abs_map = self.map(ARRAY_MAP["abs"])
        self.id_map = self.map(ARRAY_MAP["id"])

        self.add = self.zip(ARRAY_ZIP["add"])
        self.sub = self.zip(ARRAY_ZIP["sub"])
        self.mul = self.zip(ARRAY_ZIP["mul"])
        self.lt = self.zip(ARRAY_ZIP["lt"])
        self.gt = self.zip(ARRAY_ZIP["gt"])
        self.eq = self.zip(ARRAY_ZIP["eq"])
        self.pow = self.zip(ARRAY_ZIP["pow"])
        self.is_close = self.zip(ARRAY_ZIP["is_close"])
        self.relu_back = self.zip(ARRAY_ZIP["relu_back"])
        self.log_back = self.zip(ARRAY_ZIP["log_back"])
        self.inv_back = self.zip(ARRAY_ZIP["inv_back"])
        self.sigmoid_back = self.zip(ARRAY_ZIP["sigmoid_back"])
        self.sqrt_back = self.zip(ARRAY_ZIP["sqrt_back"])
        self.abs_back = self.zip(ARRAY_ZIP["abs_back"])

        self.add_reduce = self.reduce(*ARRAY_REDUCE["sum"])
        self.mul_reduce = self.reduce(*ARRAY_REDUCE["prod"])
        self.max_reduce = self.reduce(*ARRAY_REDUCE["max"])
        self.min_reduce = self.reduce(*ARRAY_REDUCE["min"])

    # ------------------------------------------------------------------
    # The six primitives
    # ------------------------------------------------------------------
    def map(self, fn: Callable[[np.ndarray], np.ndarray]):
        """
        Higher-order unary elementwise map.

        Returns a function::

            map(fn)(a: TensorData, out: Optional[TensorData] = None) -> TensorData

        If ``out`` is ``None`` a new contiguous tensor ``a.shape`` is
        allocated.  ``a`` may broadcast to ``out`` if it is smaller.
        """

        def ret(a: TensorData, out: Optional[TensorData] = None) -> TensorData:
            if out is None:
                out = TensorData.empty(a.shape, a.dtype)
            a_arr = a.numpy_view()
            if a.shape != out.shape:
                a_arr = np.broadcast_to(a_arr, out.shape)
            out.numpy_view()[...] = fn(a_arr)
            # numpy's ufunc machinery can leave sticky FPU flags that leak into
            # a later BLAS-backed matmul as misleading RuntimeWarnings; clear
            # them here so elementwise ops stay silent (PyTorch also is).
            np.seterr(over="ignore", under="ignore", divide="ignore", invalid="ignore")
            return out

        return ret

    def zip(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        """
        Higher-order binary elementwise op with broadcasting.

        Returns a function::

            zip(fn)(a: TensorData, b: TensorData) -> TensorData

        The two inputs are broadcast together (numpy rules); if either is a
        broadcast view, its stride-0 layout is exploited for free by numpy.
        """

        def ret(a: TensorData, b: TensorData) -> TensorData:
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = TensorData.empty(c_shape, a.dtype)
            a_arr = np.broadcast_to(a.numpy_view(), out.shape)
            b_arr = np.broadcast_to(b.numpy_view(), out.shape)
            out.numpy_view()[...] = fn(a_arr, b_arr)
            return out

        return ret

    def reduce(self, ufunc: np.ufunc, start: float):
        """
        Higher-order dimension reduction.

        Returns a function::

            reduce(ufunc, start)(a: TensorData, dim: int) -> TensorData

        The reduced dimension is kept with size 1 (keepdims semantics) so the
        result still broadcasts against the input in later ops.
        """

        def ret(a: TensorData, dim: int) -> TensorData:
            if not (0 <= dim < a.dims):
                raise IndexError(f"dim {dim} out of range for shape {a.shape}")
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = TensorData.empty(tuple(out_shape), a.dtype)
            reduced = ufunc.reduce(
                a.numpy_view(), axis=dim, keepdims=True, initial=start, dtype=a.dtype
            )
            out.numpy_view()[...] = reduced
            return out

        return ret

    def matmul(self, a: TensorData, b: TensorData) -> TensorData:
        """
        (Batched) matrix product of two tensors.

        Supports 2D ``(M, K) @ (K, N) -> (M, N)`` and, more generally, any
        pair of ND arrays whose leading dims are broadcastable and whose last
        two dims form an inner product (so batched attention-like matmuls
        work out of the box).  Pure vector pairs reduce to a scalar.
        """
        A, B = a.numpy_view(), b.numpy_view()

        # -- Vector x vector -> scalar (kept as shape (1,)).
        if A.ndim == 1 and B.ndim == 1:
            return TensorData(np.asarray([np.dot(A, B)], dtype=np.float32), (1,))

        # -- Promote 1-D operands so numpy can do the heavy lifting, then
        #    drop the dummy dim afterwards so results keep natural shapes.
        promote_a = A.ndim == 1
        promote_b = B.ndim == 1
        if promote_a:
            A = A[None, :]  # (K,) -> (1, K)
        if promote_b:
            B = B[..., None]  # (K,) -> (K, 1)

        # numpy's BLAS wrapper can report stale FPU exception flags left by
        # earlier vectorised kernels as if they came from this matmul, so the
        # inputs are finite while numpy warns about overflow/divide-by-zero.
        # Builders stay silent on FP flags (as PyTorch does); genuine
        # non-finite results surface as inf/nan in tensor values instead.
        with np.errstate(all="ignore"):
            C = np.matmul(A, B)  # leading batch dims broadcast for free
        if promote_a and C.ndim == 2:
            C = np.squeeze(C, axis=0)  # (1, N) -> (N,)
        if promote_b:
            C = np.squeeze(C, axis=-1)  # (..., M, 1) -> (..., M)

        return TensorData(np.ascontiguousarray(C), tuple(C.shape))

    # ------------------------------------------------------------------
    # Fused ops (mirror the `softmax_kernel.cu` / `layernorm_kernel.cu`)
    # ------------------------------------------------------------------
    #
    # These are "fused" in the CUDA sense: a single kernel pass computes the
    # whole softmax (or layer-norm) row reduction.  Here on the CPU backend
    # they are plain vectorised numpy; the numba/cuda backends provide
    # dedicated kernels with the *same* semantics:
    #
    #   * attn_softmax:  out = exp(x + m - max(x + m)) / (sum + 1e-8)
    #                    backward  d  = s * (g - sum(s * g))
    #   * layernorm:     out = (x - mean) / sqrt(var + 1e-5) * gamma + beta
    #                    backward as in the LightSeq layernorm kernels.

    #: Numeric stabilizer used in the softmax denominator (matches the .cu).
    ATTN_SOFTMAX_EPS = 1e-8

    #: Variance epsilon for layer norm (matches torchlight's ``layer_norm``).
    LAYERNORM_EPS = 1e-5

    def attn_softmax_fw(self, a: TensorData, mask: TensorData) -> TensorData:
        r"""
        Softmax over the last axis of ``a + mask``.

        Mirrors ``ker_attn_softmax``: ``mask`` must be numpy-broadcastable
        to ``a``'s shape (left-aligned, so ``(S,)``, ``(B,1,1,S)`` etc.
        against ``(B,H,T,S)`` all work).  Values under the mask are expected
        to be 0 (unmasked) or ``-inf`` (masked); the kernel subtracts the
        row max for stability and normalises with a ``1e-8`` guard.
        """
        mask_broadcast = np.broadcast_to(mask.numpy_view(), a.shape)
        z = a.numpy_view() + mask_broadcast
        mx = np.max(z, axis=-1, keepdims=True)
        e = np.exp(z - mx)
        out = e / (e.sum(axis=-1, keepdims=True) + self.ATTN_SOFTMAX_EPS)
        return TensorData(np.ascontiguousarray(out), a.shape)

    def attn_softmax_bw(self, out_grad: TensorData, soft_out: TensorData) -> TensorData:
        r"""
        Softmax backward: :math:`d = s \odot (g - \sum_j s_j g_j)` over the
        last axis, matching ``ker_attn_softmax_bw``.
        """
        g = out_grad.numpy_view()
        s = soft_out.numpy_view()
        dot = np.sum(g * s, axis=-1, keepdims=True)
        return TensorData(np.ascontiguousarray(s * (g - dot)), out_grad.shape)

    def layernorm_fw(self, a: TensorData, gamma: TensorData, beta: TensorData) -> TensorData:
        r"""
        Layer norm over the last axis.  ``gamma``/``beta`` are per-hidden-dim
        (broadcastable to ``a``).  Uses the *population* variance.
        """
        A = a.numpy_view()
        mean = A.mean(axis=-1, keepdims=True)
        var = A.var(axis=-1, keepdims=True)
        normed = (A - mean) / np.sqrt(var + self.LAYERNORM_EPS)
        out = normed * gamma.numpy_view() + beta.numpy_view()
        return TensorData(np.ascontiguousarray(out), a.shape)

    def layernorm_bw(
        self,
        out_grad: TensorData,
        a: TensorData,
        gamma: TensorData,
        beta: TensorData,
    ):
        r"""
        Layer norm backward (LightSeq ``ker_ln_bw_dgamma_dbetta`` +
        ``ker_ln_bw_dinp``):

        .. math::
            d\beta   &= \sum_{\text{rows}} g \\
            d\gamma  &= \sum_{\text{rows}} \hat{x} \odot g \\
            d\mathsf{x}_i &= \left(d\hat{x}_i
              - \frac{\sum_j d\hat{x}_j + \hat{x}_i \sum_j d\hat{x}_j \hat{x}_j}
                     {H}\right) \frac{1}{\sqrt{\mathsf{var} + \epsilon}}
        """
        G = out_grad.numpy_view()
        A = a.numpy_view()
        mean = A.mean(axis=-1, keepdims=True)
        var = A.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.LAYERNORM_EPS)
        xhat = (A - mean) * inv

        dxhat = G * gamma.numpy_view()
        reduce_axes = tuple(range(A.ndim - 1))
        dbeta = G.sum(axis=reduce_axes)
        dgamma = (G * xhat).sum(axis=reduce_axes)

        sum_dxhat = dxhat.sum(axis=-1, keepdims=True)
        sum_dxhat_xhat = (dxhat * xhat).sum(axis=-1, keepdims=True)
        dinp = (dxhat - (sum_dxhat + xhat * sum_dxhat_xhat) / A.shape[-1]) * inv

        return (
            TensorData(np.ascontiguousarray(dinp), a.shape),
            TensorData(np.ascontiguousarray(dgamma), gamma.shape),
            TensorData(np.ascontiguousarray(dbeta), beta.shape),
        )

    def allclose(self, a: TensorData, b: TensorData, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Numeric closeness check between two tensor datas."""
        return bool(np.allclose(a.numpy_view(), b.numpy_view(), rtol=rtol, atol=atol))

    # ------------------------------------------------------------------
    # Pickling
    # ------------------------------------------------------------------
    def __getstate__(self) -> tuple:
        # The bound op closures (``ret`` wrappers) are lambdas over ``self``
        # and cannot be pickled.  Drop them and rebuild on unpickle instead.
        return ()

    def __setstate__(self, state: tuple) -> None:
        self.__init__()


#: Global default backend instance shared by all CPU tensors.
default_backend = CPUBackend()


__all__ = ["CPUBackend", "default_backend"]