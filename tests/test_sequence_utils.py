"""
Tests for the variable-length-sequence utilities: masked pooling
(``masked_sum`` / ``masked_mean`` / ``masked_max``) and gradient clipping
(``nn.utils.clip_grad_norm_``).
"""
from __future__ import annotations

import numpy as np
import pytest

import torchlight as tl
from torchlight.nn import Linear, functional as F
from torchlight.nn.utils import clip_grad_norm_

try:
    import numba  # noqa: F401
    NUMBA = True
except ImportError:
    NUMBA = False

BACKENDS = ["cpu"] + (["numba"] if NUMBA else [])


class TestMaskedPooling:
    def _data(self, backend):
        rng = np.random.RandomState(3)
        x_np = rng.randn(2, 3, 4).astype(np.float32)   # (B, L, E)
        mask_np = np.array([[1., 1., 0.], [0., 1., 1.]]).astype(np.float32)
        x = tl.tensor(x_np.copy(), device=backend, requires_grad=True)
        mask = tl.tensor(mask_np.copy(), device=backend)
        return x, mask, x_np, mask_np

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_values_match_numpy(self, backend):
        x, mask, x_np, mask_np = self._data(backend)
        b = mask_np[:, :, None]  # broadcast to (B, L, 1)

        np.testing.assert_allclose(
            F.masked_sum(x, mask, 1).to_numpy(), (x_np * b).sum(axis=1), atol=1e-5
        )
        np.testing.assert_allclose(
            F.masked_mean(x, mask, 1).to_numpy(),
            (x_np * b).sum(axis=1) / b.sum(axis=1),
            atol=1e-5,
        )
        with np.errstate(invalid="ignore"):
            np.testing.assert_allclose(
                F.masked_max(x, mask, 1).to_numpy(),
                np.where(mask_np[:, :, None] == 1.0, x_np, -1e9).max(axis=1),
                atol=1e-4,
            )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_masked_rows_do_not_leak_grad(self, backend):
        x, mask, _, _ = self._data(backend)
        F.masked_mean(x, mask, 1).sum().backward()
        g = x.grad.to_numpy()
        # The (0, 2) position was masked out -> no gradient reaching it.
        np.testing.assert_allclose(g[0, 2], np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(g[1, 0], np.zeros(4), atol=1e-6)
        # unmasked positions did get the mean-gradient
        assert np.abs(g[0, 0]).sum() > 0

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_all_masked_row_is_zero_not_nan(self, backend):
        x = tl.tensor([[[1., 2.], [3., 4.]]], device=backend)   # (1, 2, 2)
        mask = tl.tensor([[0., 0.]], device=backend)             # fully masked
        out = F.masked_mean(x, mask, 1)
        np.testing.assert_allclose(out.to_numpy(), np.zeros((1, 2)), atol=1e-6)

    def test_gradcheck(self):
        from helpers import assert_allclose_grad

        rng = np.random.RandomState(5)
        x_np = rng.randn(2, 3).astype(np.float32)
        mask_np = np.array([[1., 0., 1.]], dtype=np.float32)
        mask = tl.tensor(mask_np.copy())
        x = tl.tensor(x_np.copy(), requires_grad=True)
        F.masked_mean(x, mask, 1).sum().backward()

        def fn(a):
            return F.masked_mean(a, mask, 1).sum()

        assert_allclose_grad(x, x_np, fn, name="masked_mean")


class TestClipGradNorm:
    def _small_model(self, backend):
        lin = Linear(4, 3)
        x = tl.tensor(np.ones((2, 4), np.float32), device=backend)
        return lin, x

    def test_clips_when_over_threshold(self):
        lin, x = self._small_model("cpu")
        lin(x).sum().backward()
        before = {id(p): float(p.value.grad.abs().sum().item()) for p in lin.parameters()}
        n = clip_grad_norm_(lin.parameters(), 0.01)
        assert n > 0.01, "norms before clipping should exceed the cap"
        for p in lin.parameters():
            g = p.value.grad
            assert float((g * g).sum().item()) <= 0.01 ** 2 + 1e-6

    def test_noop_when_below_threshold(self):
        lin, x = self._small_model("cpu")
        lin(x).sum().backward()
        snaps = [p.value.grad.detach().clone() for p in lin.parameters()]
        clip_grad_norm_(lin.parameters(), 1e6)
        for orig, p in zip(snaps, lin.parameters()):
            np.testing.assert_allclose(p.value.grad.to_numpy(), orig.to_numpy(), atol=1e-7)

    def test_returns_total_norm(self):
        lin, x = self._small_model("cpu")
        lin(x).sum().backward()
        total = clip_grad_norm_(lin.parameters(), 1e9, norm_type=float("inf"))
        manual = max(float(p.value.grad.abs().max().item()) for p in lin.parameters())
        assert abs(total - manual) < 1e-5

    def test_accepts_single_parameter(self):
        lin, x = self._small_model("cpu")
        lin(x).sum().backward()
        p = lin.parameters()[0]
        clip_grad_norm_(p, 0.5)
        g = p.value.grad
        assert float((g * g).sum().item()) <= 0.5 ** 2 + 1e-6

    def test_empty_parameters(self):
        assert clip_grad_norm_([], 1.0) == 0.0