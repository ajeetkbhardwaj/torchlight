"""
Tests for nn.Embedding and nn.functional.embedding (row lookup built on
IndexSelect): shapes, padding behaviour, gradient accumulation over repeated
indices, and an end-to-end context-window training loop.
"""
from __future__ import annotations

import numpy as np
import pytest

import torchlight as tl
from torchlight.nn import Embedding, Sequential, Linear, functional as F

try:
    import numba  # noqa: F401
    NUMBA = True
except ImportError:
    NUMBA = False

BACKENDS = ["cpu"] + (["numba"] if NUMBA else [])


class TestEmbeddingModule:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_lookup_shape(self, backend):
        emb = Embedding(10, 4)
        ids = tl.tensor([0, 3, 7], device=backend)
        out = emb(ids)
        assert out.shape == (3, 4)
        # rows match the weight
        np.testing.assert_allclose(
            out.to_numpy(),
            emb.weight.value.index_select(0, ids).to_numpy(),
            rtol=1e-5,
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_multidim_indices(self, backend):
        emb = Embedding(6, 3)
        ids = tl.tensor([[0, 1], [2, 3]], device=backend)
        assert emb(ids).shape == (2, 2, 3)

    def test_padding_idx_zero_and_grad(self):
        emb = Embedding(5, 3, padding_idx=0)
        ids = tl.tensor([0, 1, 0])
        out = emb(ids)
        # padding rows are exactly zero
        np.testing.assert_allclose(out[0].to_numpy(), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(out[2].to_numpy(), np.zeros(3), atol=1e-6)
        out.sum().backward()
        # no gradient flows into the padding row
        g = emb.weight.value.grad.to_numpy()
        np.testing.assert_allclose(g[0], np.zeros(3), atol=1e-6)
        # non-padding rows did receive some gradient
        assert np.abs(g[1:]).sum() > 0

    def test_params_trainable(self):
        emb = Embedding(4, 2)
        params = list(emb.parameters())
        assert len(params) == 1
        assert emb.weight.value.shape == (4, 2)

    def test_validation(self):
        with pytest.raises(ValueError):
            Embedding(0, 4)
        with pytest.raises(ValueError):
            Embedding(4, 0)
        with pytest.raises(ValueError):
            Embedding(4, 2, padding_idx=9)


class TestEmbeddingFunctional:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_dup_indices_grad_sums(self, backend):
        w = tl.tensor([[1., 2., 3.], [10., 20., 30.], [100., 200., 300.]],
                      device=backend, requires_grad=True)
        out = F.embedding(tl.tensor([0, 0, 2]), w)
        np.testing.assert_allclose(out.to_numpy(),
                                   [[1., 2., 3.], [1., 2., 3.], [100., 200., 300.]])
        out.sum().backward()
        np.testing.assert_allclose(w.grad.to_numpy(),
                                   [[2., 2., 2.], [0., 0., 0.], [1., 1., 1.]],
                                   atol=1e-5)

    def test_gradcheck(self):
        w_np = np.random.randn(3, 4).astype(np.float32)
        w = tl.tensor(w_np.copy(), requires_grad=True)
        F.embedding(tl.tensor([1, 0, 2, 1]), w).sum().backward()

        from helpers import assert_allclose_grad

        def fn(a):
            return F.embedding(tl.tensor([1, 0, 2, 1]), a).sum()

        assert_allclose_grad(w, w_np, fn, name="embedding")


class TestEmbeddingTrains:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_end_to_end(self, backend):
        # Learn to predict row-id from the sum of adjacent embedded vectors.
        rng = np.random.RandomState(0)
        vocab, d = 5, 4
        emb = Embedding(vocab, d)
        lin = Linear(d, vocab)
        params = list(emb.parameters()) + list(lin.parameters())

        dataset = [(np.array([a, b]).astype(np.float32), float(b)) for a, b in
                   [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]]

        opt = tl.optim.SGD(params, lr=0.1)
        for _ in range(400):
            opt.zero_grad()
            loss = tl.tensor(0.0)
            for x, y in dataset:
                ids = tl.tensor(x, device=backend)
                h = emb(ids).mean(dim=0)          # avg over the two tokens
                logits = lin(h.unsqueeze(0))[0]
                loss = loss + F.softmax_loss(logits, tl.tensor([y], device=backend)).sum()
            loss.backward()
            opt.step()

        # greedy decode on training points (memorisation)
        ok = 0
        for x, y in dataset:
            ids = tl.tensor(x, device=backend)
            logits = lin(emb(ids).mean(dim=0).unsqueeze(0))[0]
            pred = int(np.argmax(logits.to_numpy()))
            ok += int(pred == y)
        assert ok >= 4, f"only {ok}/5 memorised"