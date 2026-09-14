"""Tensor API behaviour tests: factories, dtypes, arithmetic, flags."""

import numpy as np
import pytest

import torchlight as tl


def test_factories_produce_expected_values_and_shapes():
    assert tl.zeros((2, 3)).shape == (2, 3)
    assert np.allclose(tl.zeros(5).to_numpy(), np.zeros(5))
    assert np.allclose(tl.ones((2, 2)).to_numpy(), np.ones((2, 2)))
    assert tl.full((2,), 3.5).to_numpy().tolist() == [3.5, 3.5]
    assert np.allclose(tl.arange(0, 5, 1).to_numpy(), np.arange(5))
    assert tl.empty((2, 3)).shape == (2, 3)
    assert tl.rand((2, 3)).shape == (2, 3)
    assert tl.randn((2, 3)).shape == (2, 3)
    # from_numpy copies data and keeps float32
    base = np.arange(6).reshape(2, 3).astype(np.float64)
    t = tl.from_numpy(base)
    assert t.dtype == np.dtype(np.float32)


def test_from_numpy_copies():
    arr = np.arange(4.0).reshape(2, 2)
    t = tl.from_numpy(arr)
    arr[0, 0] = 99.0
    assert t.to_numpy()[0, 0] == 0.0


def test_arithmetic_matches_numpy():
    a = tl.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tl.tensor([[2.0, 0.0], [1.0, 1.0]])
    A, B = a.to_numpy(), b.to_numpy()
    assert np.allclose((a + b).to_numpy(), A + B)
    assert np.allclose((a - b).to_numpy(), A - B)
    assert np.allclose((a * b).to_numpy(), A * B)
    assert np.allclose((a / b).to_numpy(), A / B)
    assert np.allclose((a @ b).to_numpy(), A @ B)
    assert np.allclose((a ** 2).to_numpy(), A ** 2)
    assert np.allclose((a + 1.0).to_numpy(), A + 1.0)


def test_comparison_ops_return_bools():
    a = tl.tensor([1.0, 2.0, 3.0])
    assert np.allclose((a < 2).to_numpy(), [1, 0, 0])
    assert np.allclose((a > 2).to_numpy(), [0, 0, 1])
    assert np.allclose((a == 2).to_numpy(), [0, 1, 0])
    assert np.allclose(a.is_close(a).to_numpy(), [1, 1, 1])


def test_reductions_match_numpy():
    x = tl.tensor(np.arange(6.0).reshape(2, 3))
    X = x.to_numpy()
    assert np.allclose(x.sum().to_numpy(), X.sum())
    assert np.allclose(x.sum(0).to_numpy(), X.sum(0))
    assert np.allclose(x.mean().to_numpy(), X.mean())
    assert np.allclose(x.max().to_numpy(), X.max())
    assert np.allclose(x.min().to_numpy(), X.min())
    assert np.allclose(x.var().to_numpy(), X.var())
    assert np.allclose(x.std().to_numpy(), X.std())


def test_requires_grad_flags_and_detach():
    x = tl.tensor([1.0, 2.0])
    assert not x.requires_grad
    x.requires_grad_()
    assert x.requires_grad and x.is_leaf()
    y = x * 2
    assert not y.is_leaf()
    z = y.detach()
    assert not z.requires_grad and not z.is_leaf() if hasattr(z, "is_leaf") else True
    # detach keeps the value
    assert np.allclose(z.to_numpy(), [2.0, 4.0])


def test_zero_grad_and_accumulation():
    x = tl.tensor([[1.0, 2.0]], requires_grad=True)
    (x * 2).mean().backward()
    first = x.grad.to_numpy().copy()
    x.zero_grad_()
    assert x.grad is None
    (x * 2).mean().backward()
    assert np.allclose(x.grad.to_numpy(), first)


def test_item_and_to_numpy():
    assert tl.tensor([3.5]).item() == pytest.approx(3.5)
    arr = np.array([[1.0, 2.0]])
    assert tl.tensor(arr).to_numpy().shape == (1, 2)


def test_clamp():
    x = tl.tensor([-1.0, 0.5, 2.0])
    assert np.allclose(x.clamp(0.0, 1.0).to_numpy(), [0.0, 0.5, 1.0])


def test_tensor_is_float32_by_default():
    assert tl.tensor(np.zeros((2,), dtype=np.float64)).dtype == np.dtype(np.float32)