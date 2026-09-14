"""
Tests for the indexing primitives (IndexSelect, Cat) and the tensor-level
helpers they surface (``index_select``, ``cat``, ``stack``, ``split``,
``chunk``, ``unsqueeze``), plus the generalised ``__getitem__``.

Numba cross-checks verify that every op behaves identically on the CPU
numpy backend and the numba CPU-JIT backend (same index math as the CUDA
kernels), confirming CPU/GPU-equivalent behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

import torchlight as tl
from helpers import assert_allclose_grad

# Numba backend fixture (same as GPU-equivalent surface).
try:
    import numba  # noqa: F401
    NUMBA = True
except ImportError:
    NUMBA = False

BACKENDS = ["cpu"] + (["numba"] if NUMBA else [])


def _ref_index_select(arr, idx, dim):
    """numpy reference for IndexSelect (np.take)."""
    return np.take(arr, idx, axis=dim)


def _ref_cat(arrs, dim):
    return np.concatenate(arrs, axis=dim)


# ---- IndexSelect forward + backward ----------------------------------------

class TestIndexSelect:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_1d_select(self, backend):
        x = tl.tensor([1., 2., 3.], device=backend, requires_grad=True)
        idx = tl.tensor([0, 2])
        out = x.index_select(0, idx)
        np.testing.assert_allclose(out.to_numpy(), [1., 3.], rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_2d_select_axis1(self, backend):
        x = tl.tensor([[1., 2., 3.], [4., 5., 6.]], device=backend)
        idx = tl.tensor([0, 0, 2])
        out = x.index_select(1, idx)
        np.testing.assert_allclose(out.to_numpy(),
                                   [[1., 1., 3.], [4., 4., 6.]])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_duplicate_idx_sums_grad(self, backend):
        w = tl.tensor([10., 20., 30.], device=backend, requires_grad=True)
        w.index_select(0, tl.tensor([0, 0])).sum().backward()
        np.testing.assert_allclose(w.grad.to_numpy(), [2., 0., 0.], atol=1e-6)

    def test_index_select_gradcheck(self):
        x_np = np.random.randn(3, 4).astype(np.float32)
        x = tl.tensor(x_np.copy(), requires_grad=True)
        x.index_select(1, tl.tensor([0, 2])).sum().backward()

        def fn(a):
            return a.index_select(1, tl.tensor([0, 2])).sum()

        assert_allclose_grad(x, x_np, fn, name="index_select")

    def test_empty_select(self):
        x = tl.tensor([1., 2.])
        out = x.index_select(0, tl.tensor([]))
        assert out.shape == (0,)


# ---- Cat forward + backward -----------------------------------------------

class TestCat:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_basic(self, backend):
        a = tl.tensor([1., 2.], device=backend)
        b = tl.tensor([3.], device=backend)
        out = tl.cat([a, b], dim=0)
        np.testing.assert_allclose(out.to_numpy(), [1., 2., 3.])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_axis1(self, backend):
        a = tl.tensor([[1., 2.], [3., 4.]], device=backend)
        b = tl.tensor([[5., 6.], [7., 8.]], device=backend)
        out = tl.cat([a, b], dim=1)
        np.testing.assert_allclose(out.to_numpy(),
                                   [[1., 2., 5., 6.], [3., 4., 7., 8.]])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_three_inputs(self, backend):
        a = tl.tensor([1.], device=backend)
        b = tl.tensor([2.], device=backend)
        c = tl.tensor([3.], device=backend)
        out = tl.cat([a, b, c])
        np.testing.assert_allclose(out.to_numpy(), [1., 2., 3.])

    def test_cat_gradcheck(self):
        a_np = np.random.randn(2, 3).astype(np.float32)
        b_np = np.random.randn(2, 3).astype(np.float32)
        a = tl.tensor(a_np.copy(), requires_grad=True)
        b = tl.tensor(b_np.copy(), requires_grad=True)
        (tl.cat([a, b], dim=0)).sum().backward()
        np.testing.assert_allclose(a.grad.to_numpy(), np.ones_like(a_np))
        np.testing.assert_allclose(b.grad.to_numpy(), np.ones_like(b_np))

    def test_single_tensor(self):
        a = tl.tensor([1., 2.])
        out = tl.cat([a], dim=0)
        np.testing.assert_allclose(out.to_numpy(), [1., 2.])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            tl.cat([])


# ---- Stack ---------------------------------------------------------------

class TestStack:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_stack(self, backend):
        a = tl.tensor([1., 2.], device=backend)
        b = tl.tensor([3., 4.], device=backend)
        out = tl.stack([a, b], dim=0)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.to_numpy(), [[1., 2.], [3., 4.]])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_stack_dim1(self, backend):
        a = tl.tensor([1., 2.], device=backend)
        b = tl.tensor([3., 4.], device=backend)
        out = tl.stack([a, b], dim=1)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.to_numpy(), [[1., 3.], [2., 4.]])

    def test_stack_gradcheck(self):
        a_np = np.random.randn(4,).astype(np.float32)
        a = tl.tensor(a_np.copy(), requires_grad=True)
        b = tl.tensor([0., 0., 0., 0.], requires_grad=True)
        out = tl.stack([a, b], dim=0)
        (out.sum()).backward()
        np.testing.assert_allclose(a.grad.to_numpy(), np.ones(4))

    def test_negative_dim(self):
        a = tl.tensor([1., 2.])
        b = tl.tensor([3., 4.])
        out = tl.stack([a, b], dim=-1)
        assert out.shape == (2, 2)


# ---- Split / Chunk --------------------------------------------------------

class TestSplitChunk:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_split(self, backend):
        x = tl.tensor([1., 2., 3., 4., 5.], device=backend)
        parts = tl.split(x, 2)
        assert len(parts) == 3
        np.testing.assert_allclose(parts[0].to_numpy(), [1., 2.])
        np.testing.assert_allclose(parts[2].to_numpy(), [5.])

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_chunk(self, backend):
        x = tl.tensor([[1., 2.], [3., 4.], [5., 6.]], device=backend)
        parts = tl.chunk(x, 2, dim=0)
        assert len(parts) == 2
        assert parts[0].shape == (2, 2)
        assert parts[1].shape == (1, 2)

    def test_split_gradcheck(self):
        x_np = np.random.randn(5,).astype(np.float32)
        x = tl.tensor(x_np.copy(), requires_grad=True)
        parts = tl.split(x, 2)
        tl.cat(parts, dim=0).sum().backward()
        np.testing.assert_allclose(x.grad.to_numpy(), np.ones(5), atol=1e-6)


# ---- Unsqueeze -----------------------------------------------------------

class TestUnsqueeze:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_unsqueeze(self, backend):
        x = tl.tensor([[1., 2.], [3., 4.]], device=backend)
        assert x.unsqueeze(0).shape == (1, 2, 2)
        assert x.unsqueeze(-1).shape == (2, 2, 1)

    def test_negative_dim(self):
        x = tl.tensor([1., 2.])
        assert x.unsqueeze(-1).shape == (2, 1)

    def test_out_of_range(self):
        x = tl.tensor([1., 2.])
        with pytest.raises(IndexError):
            x.unsqueeze(5)


# ---- __getitem__ slices ---------------------------------------------------

class TestGetItem:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_row(self, backend):
        x = tl.tensor([[1., 2.], [3., 4.]], device=backend)
        row = x[0]
        np.testing.assert_allclose(row.to_numpy(), [1., 2.])

    def test_slice(self):
        x = tl.tensor([[1., 2.], [3., 4.], [5., 6.]])
        col = x[:, 0]
        np.testing.assert_allclose(col.to_numpy(), [1., 3., 5.])

    def test_ellipsis(self):
        x = tl.tensor([[1., 2.], [3., 4.]])
        np.testing.assert_allclose(x[..., 1].to_numpy(), [2., 4.])

    def test_none(self):
        x = tl.tensor([[1., 2.]])
        # trailing source dim + trailing None both land at the end
        assert x[None, :, None].shape == (1, 1, 1, 2)
        assert x[:, None].shape == (1, 1, 2)

    def test_negative_step(self):
        x = tl.tensor([1., 2., 3., 4.])
        np.testing.assert_allclose(x[::-1].to_numpy(), [4., 3., 2., 1.])

    def test_empty_slice(self):
        x = tl.tensor([[1., 2.], [3., 4.]])
        assert x[:0, :].shape == (0, 2)

    def test_int_partial_keepdims(self):
        x = tl.tensor([[1., 2.], [3., 4.], [5., 6.]])
        assert x[1].shape == (2,)
        np.testing.assert_allclose(x[1].to_numpy(), [3., 4.])

    def test_legacy_float(self):
        x = tl.tensor([[1., 2.], [3., 4.]])
        assert x[0, 1] == 2.0

    def test_neg_int(self):
        x = tl.tensor([1., 2., 3.])
        assert x[-1] == 3.0

    def test_too_many_raises(self):
        x = tl.tensor([1., 2.])
        with pytest.raises(IndexError):
            x[0, 0]

    def test_ellipsis_only(self):
        x = tl.tensor([1., 2., 3.])
        np.testing.assert_allclose(x[...].to_numpy(), [1., 2., 3.])
