"""Functional helpers added for parity: GELU, logsumexp, one_hot,
softmax_loss, argmax, and the LayerNorm1d / GELU modules."""

import math

import numpy as np
import pytest

import torchlight as tl
from torchlight.nn import GELU, LayerNorm1d
from torchlight.nn import functional as F

from helpers import assert_allclose_grad


def _gelu_ref(x):
    """Matched `tanh`-approximation reference."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x**3)))


def test_gelu_matches_reference():
    x = tl.tensor(np.linspace(-3.0, 3.0, 13).astype(np.float32))
    got = F.gelu(x).to_numpy()
    ref = np.array([_gelu_ref(v) for v in x.to_numpy().tolist()])
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_gelu_module():
    m = GELU()
    x = tl.tensor([0.5, -0.5, 0.0])
    np.testing.assert_allclose(
        m(x).to_numpy(), F.gelu(x).to_numpy(), atol=1e-6
    )


def test_gelu_grad():
    x0 = np.linspace(-2.0, 2.0, 9).astype(np.float32)
    xg = tl.tensor(x0)
    xg.requires_grad_(True)
    F.gelu(xg).sum().backward()
    assert_allclose_grad(xg, x0, lambda t: F.gelu(t).sum())


def test_logsumexp_matches_reference():
    x = tl.tensor(np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32))
    got = F.logsumexp(x, dim=1).to_numpy()
    a = x.to_numpy()
    mx = a.max(axis=1, keepdims=True)
    ref = mx + np.log(np.exp(a - mx).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(got, ref, atol=1e-4)
    assert F.logsumexp(x, dim=0).shape == (1, 4)  # keepdims=True default


def test_logsumexp_keepdim_false():
    x = tl.tensor(np.random.default_rng(1).normal(size=(2, 5)).astype(np.float32))
    out = F.logsumexp(x, dim=1, keepdim=False)
    assert out.shape == (2,)
    np.testing.assert_allclose(
        out.to_numpy(), F.logsumexp(x, dim=1).to_numpy()[:, 0], atol=1e-6
    )


def test_logsumexp_grad():
    x0 = np.random.default_rng(2).normal(size=(2, 3)).astype(np.float32)
    xg = tl.tensor(x0)
    xg.requires_grad_(True)
    F.logsumexp(xg, dim=1).sum().backward()
    assert_allclose_grad(xg, x0, lambda t: F.logsumexp(t, dim=1).sum(), rtol=8e-3, atol=8e-3)


def test_one_hot():
    out = F.one_hot(tl.tensor([2, 0, 1]), 3)
    assert out.shape == (3, 3)
    expected = np.eye(3, dtype=np.float32)[[2, 0, 1]]
    np.testing.assert_array_equal(out.to_numpy(), expected)

    out2 = F.one_hot(tl.tensor([[0, 2], [1, 0]]), 3)
    assert out2.shape == (2, 2, 3)
    assert out2[0, 1, 2] == 1.0
    assert out2.to_numpy().sum(axis=-1).tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_one_hot_empty_and_validation():
    out = F.one_hot(tl.tensor([]), 4)
    assert out.shape == (0, 4)
    with pytest.raises(ValueError):
        F.one_hot(tl.tensor([1]), 0)


def test_softmax_loss_is_per_sample_nll():
    rng = np.random.default_rng(3)
    logits = tl.tensor(rng.normal(size=(4, 5)).astype(np.float32))
    target = tl.tensor([1, 2, 0, 4])
    loss = F.softmax_loss(logits, target)
    assert loss.shape == (4,)
    ref = -F.log_softmax(logits, dim=1)._gather_index(target, 1).contiguous().view(4)
    np.testing.assert_allclose(loss.to_numpy(), ref.to_numpy(), atol=1e-6)
    # per-sample mean equals the reduced cross-entropy
    np.testing.assert_allclose(
        loss.mean().item(), F.cross_entropy(logits, target).item(), atol=1e-6
    )


def test_argmax_is_one_hot_of_max_positions():
    x = tl.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, -1.0]])
    oh = F.argmax(x, dim=1)
    assert oh.shape == x.shape
    np.testing.assert_array_equal(
        oh.to_numpy(),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert oh.to_numpy().dtype == np.float32
    # every row of the one-hot sums to 1 (one argmax per row)
    np.testing.assert_allclose(oh.to_numpy().sum(axis=1), 1.0)


def test_layer_norm1d_equals_functional():
    dim = 8
    m = LayerNorm1d(dim)
    x = tl.tensor(np.random.default_rng(4).normal(size=(5, dim)).astype(np.float32))
    # LayerNorm1d = (x - mean)/sqrt(var + eps) * gamma + beta, same as layer_norm
    ref = F.layer_norm(x, (dim,), eps=1e-5) * m.weight.value + m.bias.value
    np.testing.assert_allclose(m(x).to_numpy(), ref.to_numpy(), atol=1e-6)
    assert m.weight.shape == (dim,)
    assert m.bias.shape == (dim,)


def test_layer_norm1d_grad_flows_to_affine():
    rng = np.random.default_rng(5)
    m = LayerNorm1d(4)
    x = tl.tensor(rng.normal(size=(3, 4)).astype(np.float32))
    m(x).sum().backward()
    assert m.weight.value.grad is not None
    assert m.bias.value.grad is not None
    # sum reduces the per-feature gain gradient to the batch size
    np.testing.assert_allclose(m.bias.value.grad.to_numpy(), np.full(4, 3.0), atol=1e-5)


# softplus ----------------------------------------------------------------
def test_softplus_matches_reference():
    x = tl.tensor(np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32))
    got = F.softplus(x).to_numpy()
    ref = np.log1p(np.exp(x.to_numpy()))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_softplus_grad():
    """Gradient of softplus(x) should equal sigmoid(x)."""
    x_np = np.array([-5.0, -0.5, 0.3, 3.0], dtype=np.float32)
    x = tl.tensor(x_np.copy(), requires_grad=True)
    F.softplus(x).sum().backward()
    sigmoid = 1.0 / (1.0 + np.exp(-x_np))
    np.testing.assert_allclose(x.grad.to_numpy(), sigmoid, atol=1e-5)