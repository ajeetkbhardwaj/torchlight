"""
Numba-accelerated backends for Torchlight.

MiniTorch-style design, adapted to the ``map / zip / reduce / matmul`` kernel
interface used everywhere else in Torchlight:

* :class:`NumbaBackend` -- CPU JIT kernels compiled once by ``numba.njit``.
  Runs everywhere numba runs; serves as the locally-testable reference for the
  CUDA kernels (same index math, same fn-id dispatch).
* :class:`CudaBackend` -- the same kernels launched as ``numba.cuda.jit``
  device kernels, mirroring the *verified* CUDA sources in
  ``torchlight/cuda_kernels/`` (``combine.cu`` plus the fused
  ``softmax_kernel.cu`` / ``layernorm_kernel.cu``, which can be exercised on a
  CUDA simulator with LeetGPU).  Requires a real NVIDIA GPU + driver.  When
  the fused kernels are compiled to ``.so`` with nvcc they are loaded through
  ctypes (like MiniTorch) and take over attention-softmax / layer-norm.

Both backends operate on the numpy-backed :class:`TensorData` layout used
throughout Torchlight: kernels receive a flat float32 storage plus int32
shape/strides arrays and do the indexing math themselves (stride-0 dims are
broadcast dims), exactly like the CUDA reference.

Scalar ops are dispatched by a small integer ``fn_id`` -- the same ids defined
in ``combine.h`` -- so the CPU-JIT kernels, the CUDA kernels, and the raw CUDA
sources all agree on semantics.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from ..core.device import Device, cpu as cpu_device
from ..core.tensor_data import TensorData, prod, shape_broadcast, strides_from_shape
from .cpu import ARRAY_MAP, ARRAY_REDUCE, ARRAY_ZIP, CPUBackend, default_backend

# ---------------------------------------------------------------------------
# fn-id constants (kept in lock-step with torchlight/cuda_kernels/combine.h)
# ---------------------------------------------------------------------------
ADD, MUL, ID, NEG, LT, EQ = 1, 2, 3, 4, 5, 6
SIGMOID, RELU, RELU_BACK, LOG, LOG_BACK = 7, 8, 9, 10, 11
EXP, INV, INV_BACK, IS_CLOSE, MAX = 12, 13, 14, 15, 16
POW, TANH, SQRT, ABS, SUB = 17, 18, 19, 20, 21
GT, SIGMOID_BACK, MIN, DIV = 22, 23, 24, 25

_MAX_DIMS = 10

#: numpy (or numpy-flavoured) callable -> fn-id, so ``map``/``zip``/``reduce``
#: can dispatch to a compiled kernel whenever they recognise the function.
_MAP_IDS: dict = {
    ARRAY_MAP[k]: v
    for k, v in {
        "id": ID,
        "neg": NEG,
        "inv": INV,
        "sigmoid": SIGMOID,
        "tanh": TANH,
        "relu": RELU,
        "log": LOG,
        "exp": EXP,
        "sqrt": SQRT,
        "abs": ABS,
    }.items()
}
_MAP_IDS.update(
    {
        np.log: LOG,
        np.exp: EXP,
        np.sqrt: SQRT,
        np.abs: ABS,
        np.negative: NEG,
        np.reciprocal: INV,
        np.tanh: TANH,
    }
)

_ZIP_IDS: dict = {
    ARRAY_ZIP[k]: v
    for k, v in {
        "add": ADD,
        "sub": SUB,
        "mul": MUL,
        "div": DIV,
        "lt": LT,
        "gt": GT,
        "eq": EQ,
        "pow": POW,
        "is_close": IS_CLOSE,
        "relu_back": RELU_BACK,
        "log_back": LOG_BACK,
        "inv_back": INV_BACK,
        "sigmoid_back": SIGMOID_BACK,
    }.items()
}
_ZIP_IDS.update(
    {
        np.add: ADD,
        np.subtract: SUB,
        np.multiply: MUL,
        np.divide: DIV,
        np.less: LT,
        np.greater: GT,
        np.equal: EQ,
        np.power: POW,
    }
)

_REDUCE_IDS: dict = {
    ARRAY_REDUCE["sum"][0]: ADD,
    ARRAY_REDUCE["prod"][0]: MUL,
    ARRAY_REDUCE["max"][0]: MAX,
    ARRAY_REDUCE["min"][0]: MIN,
}

# ---------------------------------------------------------------------------
# numba availability
# ---------------------------------------------------------------------------
try:
    import numba

    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numpy-less environment safeguard
    _HAS_NUMBA = False

if _HAS_NUMBA:
    import math

    # ------------------------------------------------------------------
    # Device-side scalar + index helpers (mirror combine.cu)
    # ------------------------------------------------------------------
    @njit()
    def _apply(fn_id, x, y):
        if fn_id == ADD:
            return x + y
        elif fn_id == MUL:
            return x * y
        elif fn_id == ID:
            return x
        elif fn_id == NEG:
            return -x
        elif fn_id == LT:
            return 1.0 if x < y else 0.0
        elif fn_id == EQ:
            return 1.0 if x == y else 0.0
        elif fn_id == SIGMOID:
            if x >= 0:
                return 1.0 / (1.0 + math.exp(-x))
            return math.exp(x) / (1.0 + math.exp(x))
        elif fn_id == RELU:
            return x if x > 0 else 0.0
        elif fn_id == RELU_BACK:
            return y if x > 0 else 0.0
        elif fn_id == LOG:
            return math.log(x + 1e-6)
        elif fn_id == LOG_BACK:
            return y / (x + 1e-6)
        elif fn_id == EXP:
            return math.exp(x)
        elif fn_id == INV:
            return 1.0 / x
        elif fn_id == INV_BACK:
            return -(1.0 / (x * x)) * y
        elif fn_id == IS_CLOSE:
            return 1.0 if abs(x - y) < 1e-2 else 0.0
        elif fn_id == MAX:
            return x if x > y else y
        elif fn_id == POW:
            return x ** y
        elif fn_id == TANH:
            return math.tanh(x)
        elif fn_id == SQRT:
            return math.sqrt(x)
        elif fn_id == ABS:
            return math.fabs(x)
        elif fn_id == SUB:
            return x - y
        elif fn_id == GT:
            return 1.0 if x > y else 0.0
        elif fn_id == SIGMOID_BACK:
            return x * (1.0 - x) * y
        elif fn_id == MIN:
            return x if x < y else y
        elif fn_id == DIV:
            return x / y
        return x + y

    @njit()
    def _index_to_position(index, strides, num_dims):
        position = 0
        for i in range(num_dims):
            position += index[i] * strides[i]
        return position

    @njit()
    def _to_index(ordinal, shape, out_index, num_dims):
        cur = ordinal
        for i in range(num_dims - 1, -1, -1):
            sh = shape[i]
            out_index[i] = cur % sh
            cur //= sh

    @njit()
    def _broadcast_index(big_index, big_shape, shape, out_index, num_dims_big, num_dims):
        for i in range(num_dims):
            if shape[i] > 1:
                out_index[i] = big_index[i + (num_dims_big - num_dims)]
            else:
                out_index[i] = 0

    # ------------------------------------------------------------------
    # CPU-JIT kernels (the FastOps twin of the CUDA kernels)
    # ------------------------------------------------------------------
    @njit()
    def _numba_map(fn_id, out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides, shape_size):
        out_index = np.empty(_MAX_DIMS, np.int32)
        in_index = np.empty(_MAX_DIMS, np.int32)
        for i in range(out_size):
            _to_index(i, out_shape, out_index, shape_size)
            _broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size)
            o = _index_to_position(out_index, out_strides, shape_size)
            j = _index_to_position(in_index, in_strides, shape_size)
            out[o] = _apply(fn_id, in_storage[j], 0.0)

    @njit()
    def _numba_zip(fn_id, out, out_shape, out_strides, out_size, out_shape_size,
                   a_storage, a_shape, a_strides, a_shape_size,
                   b_storage, b_shape, b_strides, b_shape_size):
        out_index = np.empty(_MAX_DIMS, np.int32)
        a_index = np.empty(_MAX_DIMS, np.int32)
        b_index = np.empty(_MAX_DIMS, np.int32)
        for i in range(out_size):
            _to_index(i, out_shape, out_index, out_shape_size)
            o = _index_to_position(out_index, out_strides, out_shape_size)
            _broadcast_index(out_index, out_shape, a_shape, a_index, out_shape_size, a_shape_size)
            j = _index_to_position(a_index, a_strides, a_shape_size)
            _broadcast_index(out_index, out_shape, b_shape, b_index, out_shape_size, b_shape_size)
            k = _index_to_position(b_index, b_strides, b_shape_size)
            out[o] = _apply(fn_id, a_storage[j], b_storage[k])

    @njit()
    def _numba_reduce(fn_id, out, out_shape, out_strides, out_size,
                      a_storage, a_shape, a_strides, reduce_dim, reduce_value, shape_size):
        out_index = np.empty(_MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]
        for i in range(out_size):
            _to_index(i, out_shape, out_index, shape_size)
            o = _index_to_position(out_index, out_strides, shape_size)
            acc = reduce_value
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = _index_to_position(out_index, a_strides, shape_size)
                acc = _apply(fn_id, acc, a_storage[j])
            out[o] = acc

    @njit(parallel=True)
    def _numba_matmul(out, out_shape, out_strides,
                      a_storage, a_shape, a_strides,
                      b_storage, b_shape, b_strides,
                      batch, m, n, p):
        for b in prange(batch):
            a_off = b * a_strides[0] if a_shape[0] > 1 else 0
            b_off = b * b_strides[0] if b_shape[0] > 1 else 0
            for i in prange(m):
                for j in range(p):
                    total = 0.0
                    for k in range(n):
                        total += a_storage[a_off + i * a_strides[1] + k * a_strides[2]] * b_storage[
                            b_off + k * b_strides[1] + j * b_strides[2]
                        ]
                    out[b * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = total

    # ------------------------------------------------------------------
    # Fused row kernels -- attention softmax + layer norm.
    # ------------------------------------------------------------------
    # These mirror `softmax_kernel.cu` / `layernorm_kernel.cu` semantics:
    # each kernel reduces over the *last* axis of a contiguous (rows, cols)
    # buffer.  The backend wrapper broadcasts the mask / gamma / beta to the
    # input shape on the host first, so the kernels never see broadcasting.
    #
    #   attn_softmax: out = exp(x + m - max(x + m)) / (sum + 1e-8)
    #   layernorm:    out = (x - mean) / sqrt(var + 1e-5) * gamma + beta

    @njit()
    def _numba_attn_softmax_fw(out, inp, mask, rows, cols):
        for r in range(rows):
            base = r * cols
            mval = -np.inf
            for c in range(cols):
                v = inp[base + c] + mask[base + c]
                if v > mval:
                    mval = v
            s = 0.0
            for c in range(cols):
                out[base + c] = math.exp(inp[base + c] + mask[base + c] - mval)
                s += out[base + c]
            inv = 1.0 / (s + 1e-8)
            for c in range(cols):
                out[base + c] *= inv

    @njit()
    def _numba_attn_softmax_bw(out, grad, soft, rows, cols):
        for r in range(rows):
            base = r * cols
            s = 0.0
            for c in range(cols):
                s += grad[base + c] * soft[base + c]
            for c in range(cols):
                out[base + c] = soft[base + c] * (grad[base + c] - s)

    @njit()
    def _numba_layernorm_fw(out, inp, gamma, beta, rows, cols):
        for r in range(rows):
            base = r * cols
            mean = 0.0
            for c in range(cols):
                mean += inp[base + c]
            mean /= cols
            var = 0.0
            for c in range(cols):
                d = inp[base + c] - mean
                var += d * d
            var /= cols
            inv = 1.0 / math.sqrt(var + 1e-5)
            for c in range(cols):
                out[base + c] = (inp[base + c] - mean) * inv * gamma[c] + beta[c]

    @njit()
    def _numba_layernorm_bw(dinp, dgamma, dbeta, grad, inp, gamma, rows, cols):
        for r in range(rows):
            base = r * cols
            mean = 0.0
            for c in range(cols):
                mean += inp[base + c]
            mean /= cols
            var = 0.0
            for c in range(cols):
                d = inp[base + c] - mean
                var += d * d
            var /= cols
            inv = 1.0 / math.sqrt(var + 1e-5)
            sum_dx = 0.0
            sum_dx_xhat = 0.0
            for c in range(cols):
                dx = grad[base + c] * gamma[c]
                xhat = (inp[base + c] - mean) * inv
                sum_dx += dx
                sum_dx_xhat += dx * xhat
                dgamma[c] += xhat * grad[base + c]
                dbeta[c] += grad[base + c]
            for c in range(cols):
                dx = grad[base + c] * gamma[c]
                xhat = (inp[base + c] - mean) * inv
                dinp[base + c] = (dx - (sum_dx + xhat * sum_dx_xhat) / cols) * inv

    try:
        from numba import cuda

        _HAS_CUDA_KERNELS = True
    except Exception:  # pragma: no cover
        _HAS_CUDA_KERNELS = False

    if _HAS_CUDA_KERNELS:

        @cuda.jit()
        def _cuda_map(fn_id, out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides, shape_size):
            i = cuda.grid(1)
            if i < out_size:
                out_index = cuda.local.array(_MAX_DIMS, numba.int32)
                in_index = cuda.local.array(_MAX_DIMS, numba.int32)
                _to_index(i, out_shape, out_index, shape_size)
                _broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size)
                o = _index_to_position(out_index, out_strides, shape_size)
                j = _index_to_position(in_index, in_strides, shape_size)
                out[o] = _apply(fn_id, in_storage[j], 0.0)

        @cuda.jit()
        def _cuda_zip(fn_id, out, out_shape, out_strides, out_size, out_shape_size,
                      a_storage, a_shape, a_strides, a_shape_size,
                      b_storage, b_shape, b_strides, b_shape_size):
            i = cuda.grid(1)
            if i < out_size:
                out_index = cuda.local.array(_MAX_DIMS, numba.int32)
                a_index = cuda.local.array(_MAX_DIMS, numba.int32)
                b_index = cuda.local.array(_MAX_DIMS, numba.int32)
                _to_index(i, out_shape, out_index, out_shape_size)
                o = _index_to_position(out_index, out_strides, out_shape_size)
                _broadcast_index(out_index, out_shape, a_shape, a_index, out_shape_size, a_shape_size)
                j = _index_to_position(a_index, a_strides, a_shape_size)
                _broadcast_index(out_index, out_shape, b_shape, b_index, out_shape_size, b_shape_size)
                k = _index_to_position(b_index, b_strides, b_shape_size)
                out[o] = _apply(fn_id, a_storage[j], b_storage[k])

        @cuda.jit()
        def _cuda_reduce(fn_id, out, out_shape, out_strides, out_size,
                         a_storage, a_shape, a_strides, reduce_dim, reduce_value, shape_size):
            i = cuda.grid(1)
            if i < out_size:
                out_index = cuda.local.array(_MAX_DIMS, numba.int32)
                reduce_size = a_shape[reduce_dim]
                _to_index(i, out_shape, out_index, shape_size)
                o = _index_to_position(out_index, out_strides, shape_size)
                acc = reduce_value
                for s in range(reduce_size):
                    out_index[reduce_dim] = s
                    j = _index_to_position(out_index, a_strides, shape_size)
                    acc = _apply(fn_id, acc, a_storage[j])
                out[o] = acc

        @cuda.jit()
        def _cuda_matmul(out, out_shape, out_strides,
                         a_storage, a_shape, a_strides,
                         b_storage, b_shape, b_strides,
                         batch, m, n, p):
            a_shared = cuda.shared.array(32, numba.float32)
            b_shared = cuda.shared.array(32, numba.float32)
            tx = cuda.threadIdx.x
            ty = cuda.threadIdx.y
            b = cuda.blockIdx.z
            a_off = b * a_strides[0] * (1 if a_shape[0] > 1 else 0)
            b_off = b * b_strides[0] * (1 if b_shape[0] > 1 else 0)
            row = cuda.blockIdx.y * 32 + ty
            col = cuda.blockIdx.x * 32 + tx
            total = 0.0
            n_tiles = (n + 31) // 32
            for t in range(n_tiles):
                a_row = cuda.blockIdx.y * 32 + ty
                a_col = t * 32 + tx
                if a_row < m and a_col < n:
                    a_shared[ty, tx] = a_storage[a_off + a_row * a_strides[1] + a_col * a_strides[2]]
                else:
                    a_shared[ty, tx] = 0.0
                b_row = t * 32 + ty
                b_col = cuda.blockIdx.x * 32 + tx
                if b_row < n and b_col < p:
                    b_shared[ty, tx] = b_storage[b_off + b_row * b_strides[1] + b_col * b_strides[2]]
                else:
                    b_shared[ty, tx] = 0.0
                cuda.syncthreads()
                for k in range(32):
                    total += a_shared[ty, k] * b_shared[k, tx]
                cuda.syncthreads()
            if row < m and col < p:
                out[b * out_strides[0] + row * out_strides[1] + col * out_strides[2]] = total

        # --- fused row kernels (GPU twins of the numba kernels) ----------
        # One block per row; a shared-memory tree reduces each phase.  The
        # block size is a power of two ≤ 1024 (see `_fused_threads`).

        @cuda.jit()
        def _cuda_attn_softmax_fw(out, inp, mask, rows, cols):
            r = cuda.blockIdx.x
            if r >= rows:
                return
            tx = cuda.threadIdx.x
            T = cuda.blockDim.x
            base = r * cols
            sred = cuda.shared.array(1024, numba.float32)
            sblock = cuda.shared.array(1024, numba.float32)

            lmax = -np.inf
            for c in range(tx, cols, T):
                v = inp[base + c] + mask[base + c]
                sblock[c] = v
                if v > lmax:
                    lmax = v
            sred[tx] = lmax
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    a = sred[tx]
                    b = sred[tx + offset]
                    sred[tx] = a if a > b else b
                cuda.syncthreads()
                offset //= 2
            mval = sred[0]
            cuda.syncthreads()

            lsum = 0.0
            for c in range(tx, cols, T):
                e = math.exp(sblock[c] - mval)
                sblock[c] = e
                lsum += e
            sred[tx] = lsum
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            inv = 1.0 / (sred[0] + 1e-8)
            cuda.syncthreads()
            for c in range(tx, cols, T):
                out[base + c] = sblock[c] * inv

        @cuda.jit()
        def _cuda_attn_softmax_bw(out, grad, soft, rows, cols):
            r = cuda.blockIdx.x
            if r >= rows:
                return
            tx = cuda.threadIdx.x
            T = cuda.blockDim.x
            base = r * cols
            sred = cuda.shared.array(1024, numba.float32)
            lsum = 0.0
            for c in range(tx, cols, T):
                lsum += grad[base + c] * soft[base + c]
            sred[tx] = lsum
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            total = sred[0]
            cuda.syncthreads()
            for c in range(tx, cols, T):
                out[base + c] = soft[base + c] * (grad[base + c] - total)

        @cuda.jit()
        def _cuda_layernorm_fw(out, inp, gamma, beta, rows, cols):
            r = cuda.blockIdx.x
            if r >= rows:
                return
            tx = cuda.threadIdx.x
            T = cuda.blockDim.x
            base = r * cols
            sred = cuda.shared.array(1024, numba.float32)

            acc = 0.0
            for c in range(tx, cols, T):
                acc += inp[base + c]
            sred[tx] = acc
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            mean = sred[0] / cols
            cuda.syncthreads()

            acc = 0.0
            for c in range(tx, cols, T):
                d = inp[base + c] - mean
                acc += d * d
            sred[tx] = acc
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            var = sred[0] / cols
            inv = 1.0 / math.sqrt(var + 1e-5)
            cuda.syncthreads()
            for c in range(tx, cols, T):
                out[base + c] = (inp[base + c] - mean) * inv * gamma[c] + beta[c]

        @cuda.jit()
        def _cuda_layernorm_bw(dinp, dgamma, dbeta, grad, inp, gamma, rows, cols):
            r = cuda.blockIdx.x
            if r >= rows:
                return
            tx = cuda.threadIdx.x
            T = cuda.blockDim.x
            base = r * cols
            sred = cuda.shared.array(1024, numba.float32)
            sblock = cuda.shared.array(1024, numba.float32)

            acc = 0.0
            for c in range(tx, cols, T):
                acc += inp[base + c]
            sred[tx] = acc
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            mean = sred[0] / cols
            cuda.syncthreads()

            acc = 0.0
            for c in range(tx, cols, T):
                d = inp[base + c] - mean
                acc += d * d
            sred[tx] = acc
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            var = sred[0] / cols
            inv = 1.0 / math.sqrt(var + 1e-5)
            cuda.syncthreads()

            ldx = 0.0
            for c in range(tx, cols, T):
                dx = grad[base + c] * gamma[c]
                ldx += dx
            sred[tx] = ldx
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            sum_dx = sred[0]
            cuda.syncthreads()

            ldxh = 0.0
            for c in range(tx, cols, T):
                dx = grad[base + c] * gamma[c]
                xhat = (inp[base + c] - mean) * inv
                sblock[c] = xhat
                ldxh += dx * xhat
            sred[tx] = ldxh
            cuda.syncthreads()
            offset = T // 2
            while offset > 0:
                if tx < offset:
                    sred[tx] += sred[tx + offset]
                cuda.syncthreads()
                offset //= 2
            sum_dx_xhat = sred[0]
            cuda.syncthreads()

            cnorm = 1.0 / cols
            for c in range(tx, cols, T):
                dx = grad[base + c] * gamma[c]
                dinp[base + c] = (
                    dx - (sum_dx + sblock[c] * sum_dx_xhat) * cnorm
                ) * inv
                cuda.atomic.add(dgamma, c, sblock[c] * grad[base + c])
                cuda.atomic.add(dbeta, c, grad[base + c])

else:  # pragma: no cover
    _apply = None  # type: ignore
    _HAS_CUDA_KERNELS = False


# ---------------------------------------------------------------------------
# Kernel providers
# ---------------------------------------------------------------------------
def _fused_threads(cols: int) -> int:
    """Power-of-two block size (≤ 1024) covering ``cols`` columns.

    The fused row kernels use a block-per-row shared-memory tree reduction,
    which requires a power-of-two block size.
    """
    t = 1
    while t < cols and t < 1024:
        t <<= 1
    return max(32, t)


class NumbaProvider:
    """Runs the CPU-JIT kernels directly on numpy arrays."""

    is_cpu = True
    is_cuda = False
    device: Device = cpu_device

    def map_kernel(self, out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides, in_size, shape_size, fn_id):
        _numba_map(fn_id, out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides, shape_size)

    def zip_kernel(self, out, out_shape, out_strides, out_size, out_shape_size,
                   a_storage, a_shape, a_strides, a_size, a_shape_size,
                   b_storage, b_shape, b_strides, b_size, b_shape_size, fn_id):
        _numba_zip(fn_id, out, out_shape, out_strides, out_size, out_shape_size,
                   a_storage, a_shape, a_strides, a_shape_size,
                   b_storage, b_shape, b_strides, b_shape_size)

    def reduce_kernel(self, out, out_shape, out_strides, out_size,
                      a_storage, a_shape, a_strides, reduce_dim, reduce_value, shape_size, fn_id):
        _numba_reduce(fn_id, out, out_shape, out_strides, out_size,
                      a_storage, a_shape, a_strides, reduce_dim, reduce_value, shape_size)

    def matmul_kernel(self, out, out_shape, out_strides,
                      a_storage, a_shape, a_strides,
                      b_storage, b_shape, b_strides, batch, m, p):
        n = a_shape[2]
        _numba_matmul(out, out_shape, out_strides, a_storage, a_shape, a_strides,
                      b_storage, b_shape, b_strides, batch, m, n, p)

    def attn_softmax_fw_kernel(self, out, inp, mask, rows, cols):
        _numba_attn_softmax_fw(out, inp, mask, rows, cols)

    def attn_softmax_bw_kernel(self, out, grad, soft, rows, cols):
        _numba_attn_softmax_bw(out, grad, soft, rows, cols)

    def layernorm_fw_kernel(self, out, inp, gamma, beta, rows, cols):
        _numba_layernorm_fw(out, inp, gamma, beta, rows, cols)

    def layernorm_bw_kernel(self, dinp, dgamma, dbeta, grad, inp, gamma, rows, cols):
        _numba_layernorm_bw(dinp, dgamma, dbeta, grad, inp, gamma, rows, cols)


class CudaProvider:
    """Transfers numpy buffers to the GPU and launches the numba-cuda kernels."""

    is_cpu = False
    is_cuda = True
    device: Device = Device("cuda")

    def __init__(self) -> None:
        if not _HAS_NUMBA or not _HAS_CUDA_KERNELS:
            raise RuntimeError(
                "CudaBackend requires numba-cuda. Install it with "
                "`pip install \"torchlight[cuda]\"`."
            )
        if not cuda.is_available():
            raise RuntimeError(
                "CudaBackend requires an NVIDIA GPU with a CUDA driver "
                "(numba-cuda is installed, but no driver was found)."
            )

    def map_kernel(self, out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides, in_size, shape_size, fn_id):
        threads = 32
        blocks = (out_size + threads - 1) // threads
        d_out = cuda.device_array(out_size, dtype=np.float32)
        d_in = cuda.to_device(in_storage)
        _cuda_map[blocks, threads](
            fn_id, d_out, cuda.to_device(out_shape), cuda.to_device(out_strides), out_size,
            d_in, cuda.to_device(in_shape), cuda.to_device(in_strides), shape_size,
        )
        d_out.copy_to_host(out)

    def zip_kernel(self, out, out_shape, out_strides, out_size, out_shape_size,
                   a_storage, a_shape, a_strides, a_size, a_shape_size,
                   b_storage, b_shape, b_strides, b_size, b_shape_size, fn_id):
        threads = 32
        blocks = (out_size + threads - 1) // threads
        d_out = cuda.device_array(out_size, dtype=np.float32)
        _cuda_zip[blocks, threads](
            fn_id, d_out, cuda.to_device(out_shape), cuda.to_device(out_strides), out_size, out_shape_size,
            cuda.to_device(a_storage), cuda.to_device(a_shape), cuda.to_device(a_strides), a_shape_size,
            cuda.to_device(b_storage), cuda.to_device(b_shape), cuda.to_device(b_strides), b_shape_size,
        )
        d_out.copy_to_host(out)

    def reduce_kernel(self, out, out_shape, out_strides, out_size,
                      a_storage, a_shape, a_strides, reduce_dim, reduce_value, shape_size, fn_id):
        threads = 32
        blocks = (out_size + threads - 1) // threads
        d_out = cuda.device_array(out_size, dtype=np.float32)
        _cuda_reduce[blocks, threads](
            fn_id, d_out, cuda.to_device(out_shape), cuda.to_device(out_strides), out_size,
            cuda.to_device(a_storage), cuda.to_device(a_shape), cuda.to_device(a_strides),
            reduce_dim, reduce_value, shape_size,
        )
        d_out.copy_to_host(out)

    def matmul_kernel(self, out, out_shape, out_strides,
                      a_storage, a_shape, a_strides,
                      b_storage, b_shape, b_strides, batch, m, p):
        n = a_shape[2]
        threads = (32, 32)
        grid = ((m + 31) // 32, (p + 31) // 32, batch)
        d_out = cuda.device_array((batch, m, p), dtype=np.float32)
        _cuda_matmul[grid, threads](
            d_out, cuda.to_device(out_shape), cuda.to_device(out_strides),
            cuda.to_device(a_storage), cuda.to_device(a_shape), cuda.to_device(a_strides),
            cuda.to_device(b_storage), cuda.to_device(b_shape), cuda.to_device(b_strides),
            batch, m, n, p,
        )
        d_out.copy_to_host(out.reshape(batch, m, p))

    def attn_softmax_fw_kernel(self, out, inp, mask, rows, cols):
        threads = _fused_threads(cols)
        d_out = cuda.device_array(rows * cols, dtype=np.float32)
        _cuda_attn_softmax_fw[rows, threads](
            d_out, cuda.to_device(inp), cuda.to_device(mask), rows, cols,
        )
        d_out.copy_to_host(out)

    def attn_softmax_bw_kernel(self, out, grad, soft, rows, cols):
        threads = _fused_threads(cols)
        d_out = cuda.device_array(rows * cols, dtype=np.float32)
        _cuda_attn_softmax_bw[rows, threads](
            d_out, cuda.to_device(grad), cuda.to_device(soft), rows, cols,
        )
        d_out.copy_to_host(out)

    def layernorm_fw_kernel(self, out, inp, gamma, beta, rows, cols):
        threads = _fused_threads(cols)
        d_out = cuda.device_array(rows * cols, dtype=np.float32)
        _cuda_layernorm_fw[rows, threads](
            d_out, cuda.to_device(inp), cuda.to_device(gamma), cuda.to_device(beta), rows, cols,
        )
        d_out.copy_to_host(out)

    def layernorm_bw_kernel(self, dinp, dgamma, dbeta, grad, inp, gamma, rows, cols):
        threads = _fused_threads(cols)
        d_dinp = cuda.device_array(rows * cols, dtype=np.float32)
        d_dgamma = cuda.device_array(cols, dtype=np.float32)
        d_dbeta = cuda.device_array(cols, dtype=np.float32)
        _cuda_layernorm_bw[rows, threads](
            d_dinp, d_dgamma, d_dbeta,
            cuda.to_device(grad), cuda.to_device(inp), cuda.to_device(gamma),
            rows, cols,
        )
        d_dinp.copy_to_host(dinp)
        d_dgamma.copy_to_host(dgamma)
        d_dbeta.copy_to_host(dbeta)


# ---------------------------------------------------------------------------
# Compiled CUDA C kernels (ctypes)
#
# Exactly like MiniTorch, the GPU backend can run the *fused* ops through the
# real CUDA C kernels in ``torchlight/cuda_kernels/`` (compiled with nvcc into
# ``softmax_kernel.so`` / ``layernorm_kernel.so`` on a CUDA host).  When the
# shared objects are present, :class:`CudaCKernelProvider` routes the fused
# ops through them; otherwise the numba-cuda mirror kernels are used.
# ---------------------------------------------------------------------------

_C_KERNEL_DIR = Path(__file__).resolve().parent.parent / "cuda_kernels"


def _c_kernel_lib_path(name: str) -> Optional[Path]:
    """Locate a compiled kernel ``.so`` (env override, then the package dir)."""
    override = os.environ.get("TORCHLIGHT_CUDA_KERNEL_DIR")
    roots = [Path(override)] if override else []
    roots.append(_C_KERNEL_DIR)
    for root in roots:
        for candidate in (Path(name), root / name):
            if candidate.is_file():
                return candidate
    return None


def _load_c_kernel(path: Path):
    lib = ctypes.CDLL(str(path))
    c_float_p = ctypes.POINTER(ctypes.c_float)
    stream_p = ctypes.c_void_p

    lib.launch_attn_softmax.restype = None
    lib.launch_attn_softmax.argtypes = [
        c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_bool, stream_p,
    ]
    lib.launch_attn_softmax_bw.restype = None
    lib.launch_attn_softmax_bw.argtypes = [
        c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, stream_p,
    ]
    lib.launch_layernorm.restype = None
    lib.launch_layernorm.argtypes = [
        c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_float_p,
        ctypes.c_int, ctypes.c_int, stream_p,
    ]
    lib.launch_layernorm_bw.restype = None
    lib.launch_layernorm_bw.argtypes = [
        c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, c_float_p,
        c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, stream_p,
        stream_p,
    ]
    return lib


def _ptr(arr: np.ndarray) -> ctypes._CFuncPtr:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class CudaCKernelProvider(CudaProvider):
    """GPU provider whose fused ops run the compiled CUDA C kernels.

    Loads ``softmax_kernel.so`` and ``layernorm_kernel.so`` from
    ``torchlight/cuda_kernels/`` (compiled with ``nvcc`` on a CUDA host, the
    same approach MiniTorch uses) and calls their ``launch_*`` entry points
    through ctypes.  The primitive ops (``map``/``zip``/``reduce``/``matmul``)
    still run the numba-cuda kernels.

    The fused launchers take host buffers and move data to/from the device
    themselves.  Because the forward launchers compute *in place*, the raw
    input is copied into the output buffer first; the mask passed through is
    already broadcast to the full (rows, cols) layout, which the C kernel
    consumes as a per-row mask (batch = rows, single token per block).
    """

    uses_c_kernels = True

    def __init__(self) -> None:
        super().__init__()
        soft_path = _c_kernel_lib_path("softmax_kernel.so")
        layer_path = _c_kernel_lib_path("layernorm_kernel.so")
        if soft_path is None or layer_path is None:
            missing = "softmax_kernel.so" if soft_path is None else "layernorm_kernel.so"
            raise FileNotFoundError(
                f"Compiled CUDA C kernels not found (missing {missing}). "
                f"Compile them from torchlight/cuda_kernels/ with nvcc, or "
                f"point TORCHLIGHT_CUDA_KERNEL_DIR at their directory."
            )
        self._softmax = _load_c_kernel(soft_path)
        self._layernorm = _load_c_kernel(layer_path)

    def attn_softmax_fw_kernel(self, out, inp, mask, rows, cols):
        if cols > 1024:
            return super().attn_softmax_fw_kernel(out, inp, mask, rows, cols)
        out[:] = inp[:]
        # batch=rows, nhead=1, from_len=1: each C block handles one row and
        # consumes the already-broadcast per-row mask.
        self._softmax.launch_attn_softmax(
            _ptr(out), _ptr(mask), rows, 1, 1, cols, False, None,
        )

    def attn_softmax_bw_kernel(self, out, grad, soft, rows, cols):
        if cols > 1024:
            return super().attn_softmax_bw_kernel(out, grad, soft, rows, cols)
        out[:] = grad[:]
        self._softmax.launch_attn_softmax_bw(_ptr(out), _ptr(soft), rows, cols, None)

    def layernorm_fw_kernel(self, out, inp, gamma, beta, rows, cols):
        if cols > 1024:
            return super().layernorm_fw_kernel(out, inp, gamma, beta, rows, cols)
        vars = np.empty(rows, dtype=np.float32)
        means = np.empty(rows, dtype=np.float32)
        self._layernorm.launch_layernorm(
            _ptr(out), _ptr(vars), _ptr(means), _ptr(inp), _ptr(gamma),
            _ptr(beta), rows, cols, None,
        )

    def layernorm_bw_kernel(self, dinp, dgamma, dbeta, grad, inp, gamma, rows, cols):
        if cols > 1024:
            return super().layernorm_bw_kernel(dinp, dgamma, dbeta, grad, inp, gamma, rows, cols)
        scratch_out = np.empty(rows * cols, dtype=np.float32)
        vars = np.empty(rows, dtype=np.float32)
        means = np.empty(rows, dtype=np.float32)
        # Re-run the forward to obtain vars/means (the backward needs them).
        self._layernorm.launch_layernorm(
            _ptr(scratch_out), _ptr(vars), _ptr(means), _ptr(inp), _ptr(gamma),
            _ptr(beta), rows, cols, None,
        )
        self._layernorm.launch_layernorm_bw(
            _ptr(dgamma), _ptr(dbeta), _ptr(dinp), _ptr(grad), _ptr(inp),
            _ptr(gamma), _ptr(beta), _ptr(vars), _ptr(means), rows, cols,
            None, None,
        )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class _KernelBackend:
    """
    Maps Torchlight's numpy-era primitives (``map``/``zip``/``reduce``/
    ``matmul``) onto a :class:`KernelProvider`.

    Unknown (Python) callbacks fall back to the plain-numpy CPU backend, so the
    entire operator set keeps working; recognised ops run the compiled kernels.
    """

    def __init__(self, provider: None) -> None:
        import inspect

        self._provider = provider
        self.device: Device = provider.device
        self.is_cpu = provider.is_cpu
        self.is_cuda = provider.is_cuda
        self._cpu = default_backend

        # --- Bind the named kernels (same names CPUBackend exposes). ---
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

    # -- fused ops (attention softmax / layer norm) ----------------------
    # The CPU/numba backends work directly on numpy buffers; masks, gamma
    # and beta are broadcast to the input shape *here* on the host (like the
    # CPU reference), so the kernels themselves only ever see dense rows.

    @staticmethod
    def _flat_contig(td: TensorData) -> np.ndarray:
        """Contiguous float32 copy of a TensorData (flattens leading dims)."""
        return np.ascontiguousarray(td.contiguous().storage).reshape(-1)

    @staticmethod
    def _flat_bcast(td: TensorData, shape: Tuple[int, ...]) -> np.ndarray:
        """Broadcast *mask*-style data to ``shape`` and flatten it.

        Uses native (left-aligned) numpy broadcasting matched to the CPU
        reference: ``(S,)``, ``(B,1,1,S)``, ... against ``(B,H,T,S)`` all
        flatten to the same per-row layout the kernels consume.
        """
        view = np.broadcast_to(td.numpy_view(), shape)
        return np.ascontiguousarray(view).reshape(-1)

    def attn_softmax_fw(self, a: TensorData, mask: TensorData) -> TensorData:
        shape = a.shape
        rows = prod(shape[:-1])
        cols = shape[-1]
        out = np.empty(rows * cols, dtype=np.float32)
        self._provider.attn_softmax_fw_kernel(
            out, self._flat_contig(a), self._flat_bcast(mask, shape), rows, cols,
        )
        return TensorData(out, tuple(shape))

    def attn_softmax_bw(self, out_grad: TensorData, soft_out: TensorData) -> TensorData:
        shape = out_grad.shape
        rows = prod(shape[:-1])
        cols = shape[-1]
        out = np.empty(rows * cols, dtype=np.float32)
        self._provider.attn_softmax_bw_kernel(
            out, self._flat_contig(out_grad), self._flat_contig(soft_out), rows, cols,
        )
        return TensorData(out, tuple(shape))

    def layernorm_fw(self, a: TensorData, gamma: TensorData, beta: TensorData) -> TensorData:
        shape = a.shape
        rows = prod(shape[:-1])
        cols = shape[-1]
        out = np.empty(rows * cols, dtype=np.float32)
        self._provider.layernorm_fw_kernel(
            out, self._flat_contig(a),
            self._flat_contig(gamma), self._flat_contig(beta),
            rows, cols,
        )
        return TensorData(out, tuple(shape))

    def layernorm_bw(
        self, out_grad: TensorData, a: TensorData, gamma: TensorData, beta: TensorData,
    ):
        shape = a.shape
        gamma_shape = gamma.shape
        rows = prod(shape[:-1])
        cols = shape[-1]
        dinp = np.empty(rows * cols, dtype=np.float32)
        dgamma = np.zeros(cols, dtype=np.float32)
        dbeta = np.zeros(cols, dtype=np.float32)
        self._provider.layernorm_bw_kernel(
            dinp, dgamma, dbeta,
            self._flat_contig(out_grad), self._flat_contig(a), self._flat_contig(gamma),
            rows, cols,
        )
        return (
            TensorData(dinp, tuple(shape)),
            TensorData(dgamma, tuple(gamma_shape)),
            TensorData(dbeta, tuple(gamma_shape)),
        )

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _i32(seq: Tuple[int, ...]) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(seq, dtype=np.int32))

    # -- primitives ----------------------------------------------------
    def map(self, fn: Callable[[np.ndarray], np.ndarray]):
        fn_id = _MAP_IDS.get(fn)
        if fn_id is None:
            return self._cpu.map(fn)

        def ret(a: TensorData, out: Optional[TensorData] = None) -> TensorData:
            if out is None:
                out_shape = a.shape
                out_size = prod(out_shape)
                out_storage = np.empty(out_size, dtype=np.float32)
            else:
                out_shape = out.shape
                out_size = out.size
                out_storage = out.storage
            self._provider.map_kernel(
                out_storage,
                self._i32(out_shape),
                self._i32(strides_from_shape(out_shape)),
                out_size,
                a.storage,
                self._i32(a.shape),
                self._i32(a.strides),
                len(a.storage),
                len(a.shape),
                fn_id,
            )
            if out is not None:
                return out
            return TensorData(out_storage, tuple(out_shape))

        return ret

    def zip(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        fn_id = _ZIP_IDS.get(fn)
        if fn_id is None:
            return self._cpu.zip(fn)

        def ret(a: TensorData, b: TensorData) -> TensorData:
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out_size = prod(c_shape)
            out_storage = np.empty(out_size, dtype=np.float32)
            self._provider.zip_kernel(
                out_storage,
                self._i32(c_shape),
                self._i32(strides_from_shape(c_shape)),
                out_size,
                len(c_shape),
                a.storage,
                self._i32(a.shape),
                self._i32(a.strides),
                len(a.storage),
                len(a.shape),
                b.storage,
                self._i32(b.shape),
                self._i32(b.strides),
                len(b.storage),
                len(b.shape),
                fn_id,
            )
            return TensorData(out_storage, tuple(c_shape))

        return ret

    def reduce(self, ufunc: np.ufunc, start: float):
        fn_id = _REDUCE_IDS.get(ufunc)
        if fn_id is None:
            return self._cpu.reduce(ufunc, start)

        def ret(a: TensorData, dim: int) -> TensorData:
            if not (0 <= dim < a.dims):
                raise IndexError(f"dim {dim} out of range for shape {a.shape}")
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out_size = prod(out_shape)
            out_storage = np.empty(out_size, dtype=np.float32)
            self._provider.reduce_kernel(
                out_storage,
                self._i32(out_shape),
                self._i32(strides_from_shape(tuple(out_shape))),
                out_size,
                a.storage,
                self._i32(a.shape),
                self._i32(a.strides),
                dim,
                float(start),
                len(a.shape),
                fn_id,
            )
            return TensorData(out_storage, tuple(out_shape))

        return ret

    def matmul(self, a: TensorData, b: TensorData) -> TensorData:
        """(Batched) matmul, mirroring CPUBackend semantics over the kernel."""
        # Pure vector pairs / any 1-D operand: numpy handles the corner cases
        # (dot shapes, dummy-dim squeezing) that the 3-D kernel doesn't express.
        if a.dims == 1 or b.dims == 1:
            return self._cpu.matmul(a, b)

        both_2d = 0
        if a.dims == 2:
            a = a.reshape((1,) + a.shape)
            both_2d += 1
        if b.dims == 2:
            b = b.reshape((1,) + b.shape)
            both_2d += 1
        both_2d = both_2d == 2

        final_shape_leading = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        final_shape_leading.append(a.shape[-2])
        final_shape_leading.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2], "inner dims must match"
        out = TensorData.zeros(tuple(final_shape_leading))

        # The kernel runs a single collapsed batch dim.  The collapse is only a
        # flat reinterpretation when each operand's leading dims either reduce
        # to the broadcast batch size or to 1 (broadcast); otherwise fall back
        # to numpy, which broadcasts any ND shapes natively.
        a_lead, b_lead = a.shape[:-2], b.shape[:-2]
        batch_lead = prod(shape_broadcast(a_lead, b_lead))
        if (prod(a_lead) not in (1, batch_lead)) or (prod(b_lead) not in (1, batch_lead)):
            return self._cpu.matmul(a, b)

        # Collapse ND operands to 3-D (batch, m, n).
        m, p = a.shape[-2], b.shape[-1]
        if a.dims > 3:
            a = a.contiguous().reshape((prod(a_lead), m, a.shape[-1]))
        if b.dims > 3:
            b = b.contiguous().reshape((prod(b_lead), b.shape[-2], p))

        # Bring ``out`` to 3-D so the kernel sees (batch, rows, cols).
        if out.dims == 2:
            batch, out_shape3 = 1, (1, m, p)
        elif out.dims > 3:
            out_shape3 = (batch_lead, m, p)
            out = out.reshape(out_shape3)
        else:
            out_shape3 = out.shape
            batch = out_shape3[0]
        batch, m_out, p_out = out_shape3
        assert m_out == m and p_out == p

        out_storage = np.empty(batch * m * p, dtype=np.float32)
        self._provider.matmul_kernel(
            out_storage,
            self._i32(out_shape3),
            self._i32(strides_from_shape(out_shape3)),
            a.storage,
            self._i32(a.shape),
            self._i32(a.strides),
            b.storage,
            self._i32(b.shape),
            self._i32(b.strides),
            batch,
            m,
            p,
        )
        out = TensorData(out_storage, tuple(out_shape3))

        if both_2d:
            return out.reshape((m, p))
        if len(final_shape_leading) > out.dims:
            return out.reshape(tuple(final_shape_leading))
        return out

    def allclose(self, a: TensorData, b: TensorData, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return bool(np.allclose(a.numpy_view(), b.numpy_view(), rtol=rtol, atol=atol))

    # -- pickling -------------------------------------------------------
    def __getstate__(self) -> tuple:
        # The bound op closures (``ret`` wrappers) are lambdas over ``self``
        # and cannot be pickled.  Drop everything and rebuild on unpickle.
        return ()

    def __setstate__(self, state: tuple) -> None:
        self.__init__()


class NumbaBackend(_KernelBackend):
    """CPU-JIT numba backend (device ``'numba'``)."""

    def __init__(self) -> None:
        if not _HAS_NUMBA:
            raise RuntimeError(
                "NumbaBackend requires numba. It is an internal CPU twin of "
                "CudaBackend (used to validate the GPU kernels without a GPU); "
                "CPU users should use the default numpy CPUBackend."
            )
        super().__init__(NumbaProvider())


class CudaBackend(_KernelBackend):
    """numba.cuda backend (device ``'cuda'``).

    Fused ops (attention-softmax / layer-norm) run through the compiled CUDA C
    kernels in ``torchlight/cuda_kernels/`` when their ``.so`` files are
    present, otherwise through the numba-cuda mirror kernels.
    """

    def __init__(self) -> None:
        super().__init__(_make_cuda_provider())


def _make_cuda_provider():
    """Build the CUDA provider, preferring the compiled C kernels."""
    provider = CudaProvider()
    try:
        return CudaCKernelProvider()
    except FileNotFoundError:
        return provider


#: Lazily-created singletons used by ``torchlight.backends.get_backend``.
_numba_backend: Optional[NumbaBackend] = None
_cuda_backend: Optional[CudaBackend] = None


def numba_backend() -> NumbaBackend:
    global _numba_backend
    if _numba_backend is None:
        _numba_backend = NumbaBackend()
    return _numba_backend


def cuda_backend() -> CudaBackend:
    global _cuda_backend
    if _cuda_backend is None:
        _cuda_backend = CudaBackend()
    return _cuda_backend


__all__ = ["NumbaBackend", "CudaBackend", "numba_backend", "cuda_backend", "_HAS_NUMBA"]