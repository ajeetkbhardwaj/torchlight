"""
Fused attention-softmax + layer-norm tests (Tensor-level API).

Exercises ``Tensor.attn_softmax`` / ``Tensor.layernorm`` on the numpy CPU
backend and, when numba is installed, on the numba backend (the local twin of
the CUDA kernels).  Verifies forward values and analytic-vs-numeric gradients.

Usage::

    PYTHONPATH=src pytest tests/test_fused_ops.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

import torchlight as tl

try:
    from torchlight.backends.cuda import NumbaBackend, _HAS_NUMBA
except Exception:  # pragma: no cover - numba absent
    _HAS_NUMBA = False


BACKENDS = {"cpu": tl.get_backend("cpu")}
if _HAS_NUMBA:
    BACKENDS["numba"] = tl.get_backend("numba")


def _tensors_on(backend, shape, requires_grad=True, scale=1.0, shift=0.0):
    """Return a random float32 tensor graph seeded on ``backend``."""
    data = (np.random.rand(*shape).astype(np.float32) * scale + shift)
    return tl.tensor(data, requires_grad=requires_grad, backend=backend)


# ---------------------------------------------------------------------------
# attention softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", list(BACKENDS.values()), ids=list(BACKENDS.keys()))
def test_attn_softmax_rowsums_and_shape(backend):
    a = _tensors_on(backend, (2, 3, 8), scale=4.0)
    mask = _tensors_on(backend, (8,), requires_grad=False)
    out = a.attn_softmax(mask)
    assert out.shape == a.shape
    rows = out.to_numpy()
    np.testing.assert_allclose(rows.sum(axis=-1), np.ones(a.shape[:-1]), atol=1e-4)


@pytest.mark.parametrize("backend", list(BACKENDS.values()), ids=list(BACKENDS.keys()))
def test_attn_softmax_matches_reference_formula(backend):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 1, 5, 9)).astype(np.float32)
    mask = rng.standard_normal((2, 1, 1, 9)).astype(np.float32)
    a = tl.tensor(X, backend=backend)
    m = tl.tensor(mask, backend=backend)
    z = X + mask
    ref = np.exp(z - z.max(axis=-1, keepdims=True))
    ref = ref / (ref.sum(axis=-1, keepdims=True) + 1e-8)
    np.testing.assert_allclose(a.attn_softmax(m).to_numpy(), ref, atol=1e-5)


@pytest.mark.parametrize("backend", list(BACKENDS.values()), ids=list(BACKENDS.keys()))
def test_attn_softmax_backward_gradient(backend):
    a = _tensors_on(backend, (2, 3, 6), scale=3.0)
    mask = _tensors_on(backend, (6,), requires_grad=False)
    s = a.attn_softmax(mask)
    loss = (s * s).sum()
    loss.backward()

    ref_a = a.to_numpy()
    z = ref_a + mask.to_numpy()
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    soft = e / (e.sum(axis=-1, keepdims=True) + 1e-8)
    # d/dx sum(s^2) = sum_c 2s_c (g_c - m) with g_c = 2s_c, m = sum 2s s
    m = np.sum(2.0 * soft * soft, axis=-1, keepdims=True)
    expected = soft * (2.0 * soft - m)
    np.testing.assert_allclose(a.grad.to_numpy(), expected, atol=2e-4)


@pytest.mark.parametrize("backend", list(BACKENDS.values()), ids=list(BACKENDS.keys()))
def test_attn_softmax_mask_is_constant(backend):
    a = _tensors_on(backend, (2, 4), requires_grad=True)
    m = tl.tensor(np.random.rand(4).astype(np.float32), requires_grad=True, backend=backend)
    s = a.attn_softmax(m)
    s.sum().backward()
    assert m.grad is not None
    np.testing.assert_allclose(m.grad.to_numpy(), np.zeros((4,)), atol=1e-6)


# ---------------------------------------------------------------------------
# layer norm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", list(BACKENDS.values()), ids=list(BACKENDS.keys()))
def test_layernorm_matches_formula_and_stats(backend):
    X = np.random.rand(4, 17).astype(np.float32) * 2.0 + 1.0
    gamma = np.random.rand(17).astype(np.float32) * 0.5 + 0.75
    beta = np.random.rand(17).astype(np.float32)
    a = tl.tensor(X, backend=backend)
    g = tl.tensor(gamma, backend=backend)
    b = tl.tensor(beta, backend=backend)

    y = a.layernorm(g, b).to_numpy()
    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    ref = (X - mean) / np.sqrt(var + 1e-5) * gamma + beta
    np.testing.assert_allclose(y, ref, atol=1e-5)


@pytest.mark.parametrize("backend", list(BACKENDS.values()), ids=list(BACKENDS.keys()))
def test_layernorm_gradients_match_numeric(backend):
    X = np.random.rand(3, 8).astype(np.float32)
    gamma = np.random.rand(8).astype(np.float32) + 0.5
    beta = np.random.rand(8).astype(np.float32)

    # gradient w.r.t. input
    a = tl.tensor(X, requires_grad=True, backend=backend)
    g = tl.tensor(gamma, backend=backend)
    b = tl.tensor(beta, backend=backend)
    a.layernorm(g, b).sum().backward()

    mean = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + 1e-5)
    xhat = (X - mean) * inv
    dxhat = np.ones_like(X) * gamma
    sum_dx = dxhat.sum(-1, keepdims=True)
    sum_dxh = (dxhat * xhat).sum(-1, keepdims=True)
    expected = (dxhat - (sum_dx + xhat * sum_dxh) / 8.0) * inv
    np.testing.assert_allclose(a.grad.to_numpy(), expected, atol=2e-4)

    # gradients w.r.t. gamma and beta
    a2 = tl.tensor(X, backend=backend)
    g2 = tl.tensor(gamma, requires_grad=True, backend=backend)
    b2 = tl.tensor(beta, requires_grad=True, backend=backend)
    a2.layernorm(g2, b2).sum().backward()
    np.testing.assert_allclose(g2.grad.to_numpy(), xhat.sum(axis=0), atol=2e-4)
    np.testing.assert_allclose(b2.grad.to_numpy(), np.full((8,), 3.0), atol=2e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))