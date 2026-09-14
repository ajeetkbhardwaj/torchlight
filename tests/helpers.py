"""Shared numeric gradient-check helper used by the autograd tests."""

from __future__ import annotations

from typing import Callable

import numpy as np

import torchlight as tl


def finite_difference_grad(fn: Callable, X0, eps: float = 4e-4):
    """
    Central-difference gradient of a scalar-valued function.

    ``fn`` must accept a numpy array and return a scalar.  Perturbations are
    applied in float64 *before* casting to the engine's float32, and a fairly
    large eps keeps the finite-difference noise below float32's roundoff.
    """
    X0 = np.asarray(X0, dtype=np.float32)
    g = np.zeros(X0.shape, dtype=np.float64)
    it = np.nditer(X0, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        Xp = X0.astype(np.float64).copy()
        Xm = X0.astype(np.float64).copy()
        Xp[i] += eps
        Xm[i] -= eps
        g[i] = (fn(Xp) - fn(Xm)) / (2 * eps)
        it.iternext()
    return g


def assert_allclose_grad(x: tl.Tensor, x_np, fn, rtol=5e-3, atol=5e-3, name=""):
    """Run f(x).backward(), then compare the accumulated grad to finite diff."""
    x_np = np.asarray(x_np, dtype=np.float32)
    assert x.grad is not None, f"{name}: backward did not accumulate a gradient"
    got = x.grad.to_numpy()
    expected = finite_difference_grad(lambda X: float(fn(tl.tensor(np.asarray(X, np.float32))).to_numpy().reshape(-1)[0]), x_np)
    np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol, err_msg=f"{name} gradcheck")