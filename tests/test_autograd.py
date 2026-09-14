"""Numeric gradchecks for the autograd engine (vs central differences)."""

import numpy as np
import pytest

import torchlight as tl
from torchlight.nn import functional as F
from helpers import assert_allclose_grad

rng = np.random.default_rng(0)
POS = rng.uniform(0.5, 1.5, (2, 3))          # strictly positive  (log/sqrt)
SIG_NEG = rng.uniform(-3.0, 3.0, (3, 4))     # mixed sign         (tanh/sigmoid/relu)
SIG_ALL = rng.uniform(-1.0, 1.0, (2, 3))


@pytest.mark.parametrize("name,fn,data", [
    ("exp", lambda t: t.exp().sum(), POS),
    ("log", lambda t: t.log().sum(), POS),
    ("sqrt", lambda t: t.sqrt().sum(), POS),
    ("sigmoid", lambda t: t.sigmoid().sum(), SIG_ALL),
    ("tanh", lambda t: t.tanh().sum(), SIG_ALL),
    ("relu", lambda t: t.relu().sum(), SIG_ALL),
    ("abs", lambda t: abs(t).sum(), SIG_ALL),
    ("pow3", lambda t: (t ** 3).sum(), SIG_ALL),
    ("neg", lambda t: (-t).sum(), SIG_ALL),
    ("div", lambda t: (t / 2.0).sum(), SIG_ALL),
])
def test_unary_grads(name, fn, data):
    x = tl.tensor(np.asarray(data, np.float32), requires_grad=True)
    fn(x).backward()
    assert_allclose_grad(x, data, fn, name=name)


@pytest.mark.parametrize("name,fn,data", [
    ("sum", lambda t: t.sum(), SIG_ALL),
    ("mean", lambda t: t.mean(), SIG_ALL),
    ("var", lambda t: t.var(), SIG_ALL),
    ("std", lambda t: t.std(), SIG_ALL),
    ("max", lambda t: t.max(), SIG_ALL),
    ("min", lambda t: t.min(), SIG_ALL),
])
def test_reduction_grads(name, fn, data):
    x = tl.tensor(np.asarray(data, np.float32), requires_grad=True)
    fn(x).backward()
    assert_allclose_grad(x, data, fn, name=name)


def test_matmul_grads_both_sides():
    A = rng.uniform(0.5, 1.5, (2, 2))
    B = rng.uniform(0.5, 1.5, (2, 2))

    a = tl.tensor(np.asarray(A, np.float32), requires_grad=True)
    (a @ tl.tensor(B)).sum().backward()
    assert_allclose_grad(a, A, lambda t: (t @ tl.tensor(B)).sum(), name="matmul-left")

    b = tl.tensor(np.asarray(B, np.float32), requires_grad=True)
    (tl.tensor(A) @ b).sum().backward()
    assert_allclose_grad(b, B, lambda t: (tl.tensor(A) @ t).sum(), name="matmul-right")


def test_shared_parent_grad_accumulates():
    # x appears on both sides of a matmul: both paths must accumulate.
    X = rng.uniform(0.5, 1.5, (2, 2))
    x = tl.tensor(np.asarray(X, np.float32), requires_grad=True)
    (x @ x).sum().backward()
    assert_allclose_grad(x, X, lambda t: (t @ t).sum(), name="matmul-shared")


def test_broadcast_grads_sum_over_broadcast_dims():
    M = rng.uniform(-1, 1, (4, 3))
    v = rng.uniform(-1, 1, (3,))
    m = tl.tensor(np.asarray(M, np.float32), requires_grad=True)
    vec = tl.tensor(np.asarray(v, np.float32), requires_grad=True)
    (m + vec).sum().backward()
    assert_allclose_grad(m, M, lambda t: (t + tl.tensor(v)).sum(), name="row-bcast-m")
    # The vector-side grad must equal the column sums of ones.
    assert np.allclose(vec.grad.to_numpy(), np.full(3, 4.0))


def test_layout_ops_grad_flows_through():
    data = np.arange(6).reshape(2, 3).astype(np.float32)
    x = tl.tensor(np.asarray(data, np.float32), requires_grad=True)
    x.view(3, 2).sum().backward()
    assert np.allclose(x.grad.to_numpy(), np.ones((2, 3)))

    y = tl.tensor(np.asarray(data, np.float32), requires_grad=True)
    y.permute(1, 0).sum().backward()
    assert np.allclose(y.grad.to_numpy(), np.ones((2, 3)))


def test_softmax_and_log_softmax_grads():
    data = rng.uniform(-1, 1, (4, 3))
    x = tl.tensor(np.asarray(data, np.float32), requires_grad=True)
    F.log_softmax(x).sum().backward()
    assert_allclose_grad(x, data, lambda t: F.log_softmax(t).sum(), name="log_softmax")


def test_deep_chain_combines_many_ops():
    data = rng.uniform(0.5, 1.2, (3, 3))
    fn = lambda t: ((t * 0.5).sigmoid() * (t - 1.0)).sum() ** 2  # noqa: E731
    x = tl.tensor(np.asarray(data, np.float32), requires_grad=True)
    fn(x).backward()
    assert_allclose_grad(x, data, fn, name="chain")