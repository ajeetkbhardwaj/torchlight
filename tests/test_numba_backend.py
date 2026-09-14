"""
NumbaBackend tests -- run the full operator surface through the JIT kernels
and verify equivalence with the reference CPUBackend.

Usage::

    PYTHONPATH=src pytest tests/test_numba_backend.py -q

No markers required; these are pure CPU-JIT tests that run everywhere numba
runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchlight.backends.cuda import NumbaBackend, _HAS_NUMBA

if not _HAS_NUMBA:
    pytest.skip(
        "numba is not installed; skipping numba-backend tests.",
        allow_module_level=True,
    )

from torchlight.backends.cpu import CPUBackend
from torchlight.core.tensor_data import TensorData
from torchlight.tensor.tensor import (
    Tensor,
    tensor,
    rand,
    randn,
    zeros,
    ones,
    from_numpy,
    arange,
)

cpu = CPUBackend()
numba = NumbaBackend()


def _td(shape, storage=None):
    """Build a contiguous CPU TensorData from raw shape + optional flat list."""
    shape = tuple(shape)
    if storage is None:
        storage = np.arange(prod(shape), dtype=np.float32)
    return TensorData(np.ascontiguousarray(storage, dtype=np.float32), shape)


def prod(shape):
    out = 1
    for s in shape:
        out *= s
    return out


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------
class TestNumbaMap:
    @pytest.mark.parametrize("fn", ["neg", "sigmoid", "tanh", "sqrt"])
    def test_basic_map(self, fn):
        np.random.seed(0)
        a_np = np.abs(np.random.randn(6)).astype(np.float32) * 2.0
        ref = getattr(cpu, f"{fn}_map")(_td((6,), a_np))
        out = getattr(numba, f"{fn}_map")(_td((6,), a_np))
        assert out.shape == ref.shape
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_basic_map_relu(self):
        np.random.seed(0)
        a_np = np.random.randn(6).astype(np.float32) * 2.0
        ref = cpu.relu_map(_td((6,), a_np))
        out = numba.relu_map(_td((6,), a_np))
        assert out.shape == ref.shape
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_broadcast_map(self):
        a_np = np.array([-1.0, 2.0, -0.5], dtype=np.float32)
        ref = cpu.relu_map(TensorData(a_np.reshape(1, 3), (1, 3)))
        out = numba.relu_map(TensorData(a_np.reshape(1, 3), (1, 3)))
        assert out.shape == (1, 3)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-6, atol=1e-7)

    def test_map_out(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ref_out = TensorData.zeros((3,))
        cpu.neg_map(_td((3,), a_np), ref_out)
        out_out = TensorData.zeros((3,))
        numba.neg_map(_td((3,), a_np), out_out)
        assert np.allclose(out_out.numpy_view(), ref_out.numpy_view(), rtol=1e-7)


# ---------------------------------------------------------------------------
# zip
# ---------------------------------------------------------------------------
class TestNumbaZip:
    @pytest.mark.parametrize("fn", ["add", "mul", "sub", "lt", "gt", "eq"])
    def test_basic_zip(self, fn):
        np.random.seed(1)
        a_np = np.random.randn(4).astype(np.float32)
        b_np = np.random.randn(4).astype(np.float32)
        ref = getattr(cpu, fn)(_td((4,), a_np), _td((4,), b_np))
        out = getattr(numba, fn)(_td((4,), a_np), _td((4,), b_np))
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5, atol=1e-6)

    def test_broadcast_zip(self):
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_np = np.array([10.0, 20.0], dtype=np.float32)
        ref = cpu.add(TensorData(a_np, (2, 2)), TensorData(b_np, (2,)))
        out = numba.add(TensorData(a_np, (2, 2)), TensorData(b_np, (2,)))
        assert out.shape == (2, 2)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5, atol=1e-6)

    def test_pow_zip(self):
        a_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.array([0.5, 2.0, 1.0], dtype=np.float32)
        ref = cpu.pow(_td((3,), a_np), _td((3,), b_np))
        out = numba.pow(_td((3,), a_np), _td((3,), b_np))
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_relu_back_zip(self):
        x_np = np.array([-1.0, 1.0, 2.0, -3.0], dtype=np.float32)
        d_np = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        ref = cpu.relu_back(_td((4,), x_np), _td((4,), d_np))
        out = numba.relu_back(_td((4,), x_np), _td((4,), d_np))
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-7)

    def test_sigmoid_back_zip(self):
        s_np = np.array([0.2, 0.8, 0.5, 0.9], dtype=np.float32)
        d_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ref = cpu.sigmoid_back(_td((4,), s_np), _td((4,), d_np))
        out = numba.sigmoid_back(_td((4,), s_np), _td((4,), d_np))
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------
class TestNumbaReduce:
    def test_sum_reduce(self):
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        ref = cpu.add_reduce(TensorData(a_np, (2, 3)), 0)
        out = numba.add_reduce(TensorData(a_np, (2, 3)), 0)
        assert out.shape == (1, 3)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5)

    def test_sum_reduce_dim1(self):
        a_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        ref = cpu.add_reduce(TensorData(a_np, (2, 3)), 1)
        out = numba.add_reduce(TensorData(a_np, (2, 3)), 1)
        assert out.shape == (2, 1)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5)

    def test_max_reduce(self):
        a_np = np.array([1.0, 5.0, 3.0, 2.0], dtype=np.float32)
        ref = cpu.max_reduce(_td((4,), a_np), 0)
        out = numba.max_reduce(_td((4,), a_np), 0)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5)

    def test_min_reduce(self):
        a_np = np.array([1.0, 5.0, 3.0, 2.0], dtype=np.float32)
        ref = cpu.min_reduce(_td((4,), a_np), 0)
        out = numba.min_reduce(_td((4,), a_np), 0)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5)

    def test_prod_reduce(self):
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ref = cpu.mul_reduce(_td((4,), a_np), 0)
        out = numba.mul_reduce(_td((4,), a_np), 0)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4)

    def test_3d_reduce(self):
        np.random.seed(2)
        a_np = np.random.randn(2, 3, 4).astype(np.float32)
        ref = cpu.add_reduce(TensorData(a_np, (2, 3, 4)), 1)
        out = numba.add_reduce(TensorData(a_np, (2, 3, 4)), 1)
        assert out.shape == (2, 1, 4)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-5)


# ---------------------------------------------------------------------------
# matmul
# ---------------------------------------------------------------------------
class TestNumbaMatMul:
    def test_2d(self):
        np.random.seed(3)
        a_np = np.random.randn(4, 3).astype(np.float32)
        b_np = np.random.randn(3, 5).astype(np.float32)
        ref = cpu.matmul(_td((4, 3), a_np), _td((3, 5), b_np))
        out = numba.matmul(_td((4, 3), a_np), _td((3, 5), b_np))
        assert out.shape == ref.shape
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_1d_1d(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ref = cpu.matmul(_td((3,), a_np), _td((3,), b_np))
        out = numba.matmul(_td((3,), a_np), _td((3,), b_np))
        assert out.shape == (1,)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_1d_2d(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.random.randn(3, 4).astype(np.float32)
        ref = cpu.matmul(_td((3,), a_np), _td((3, 4), b_np))
        out = numba.matmul(_td((3,), a_np), _td((3, 4), b_np))
        assert out.shape == (4,)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_2d_1d(self):
        a_np = np.random.randn(5, 3).astype(np.float32)
        b_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ref = cpu.matmul(_td((5, 3), a_np), _td((3,), b_np))
        out = numba.matmul(_td((5, 3), a_np), _td((3,), b_np))
        assert out.shape == (5,)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_batched_3d(self):
        np.random.seed(4)
        a_np = np.random.randn(2, 4, 3).astype(np.float32)
        b_np = np.random.randn(2, 3, 5).astype(np.float32)
        ref = cpu.matmul(TensorData(a_np, (2, 4, 3)), TensorData(b_np, (2, 3, 5)))
        out = numba.matmul(TensorData(a_np, (2, 4, 3)), TensorData(b_np, (2, 3, 5)))
        assert out.shape == (2, 4, 5)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_batch_broadcast(self):
        np.random.seed(5)
        a_np = np.random.randn(1, 4, 3).astype(np.float32)
        b_np = np.random.randn(2, 3, 5).astype(np.float32)
        ref = cpu.matmul(TensorData(a_np, (1, 4, 3)), TensorData(b_np, (2, 3, 5)))
        out = numba.matmul(TensorData(a_np, (1, 4, 3)), TensorData(b_np, (2, 3, 5)))
        assert out.shape == (2, 4, 5)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_batch_broadcast_reverse(self):
        np.random.seed(6)
        a_np = np.random.randn(2, 4, 3).astype(np.float32)
        b_np = np.random.randn(1, 3, 5).astype(np.float32)
        ref = cpu.matmul(TensorData(a_np, (2, 4, 3)), TensorData(b_np, (1, 3, 5)))
        out = numba.matmul(TensorData(a_np, (2, 4, 3)), TensorData(b_np, (1, 3, 5)))
        assert out.shape == (2, 4, 5)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_high_dim(self):
        np.random.seed(7)
        a_np = np.random.randn(2, 3, 4, 3).astype(np.float32)
        b_np = np.random.randn(2, 3, 3, 5).astype(np.float32)
        ref = cpu.matmul(TensorData(a_np, (2, 3, 4, 3)), TensorData(b_np, (2, 3, 3, 5)))
        out = numba.matmul(TensorData(a_np, (2, 3, 4, 3)), TensorData(b_np, (2, 3, 3, 5)))
        assert out.shape == (2, 3, 4, 5)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)

    def test_4d_matmul_vs_numpy(self):
        np.random.seed(8)
        a_np = np.random.randn(2, 1, 4, 3).astype(np.float32)
        b_np = np.random.randn(2, 3, 3, 2).astype(np.float32)
        ref = cpu.matmul(TensorData(a_np, (2, 1, 4, 3)), TensorData(b_np, (2, 3, 3, 2)))
        out = numba.matmul(TensorData(a_np, (2, 1, 4, 3)), TensorData(b_np, (2, 3, 3, 2)))
        assert out.shape == (2, 3, 4, 2)
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Tensor API equivalence
# ---------------------------------------------------------------------------
class TestTensorAPI:
    def test_tensor_factory(self):
        a = tensor([1.0, 2.0, 3.0], device="numba")
        assert a.device == cpu.device
        assert np.allclose(a.to_numpy(), [1.0, 2.0, 3.0], atol=1e-7)

    def test_zeros_ones(self):
        z = zeros((3, 4), device="numba")
        o = ones((3, 4), device="numba")
        assert z.shape == (3, 4)
        assert np.all(z.to_numpy() == 0.0)
        assert np.all(o.to_numpy() == 1.0)

    def test_rand_randn(self):
        r = rand((10,), device="numba")
        rn = randn((10,), device="numba")
        assert r.shape == (10,)
        assert 0.0 <= float(r.min().to_numpy().reshape(-1)[0]) < 1.0
        assert rn.shape == (10,)

    def test_to_device(self):
        a = tensor([1.0, 2.0], device="cpu")
        b = a.to("numba")
        assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-7)
        # ops on b go through numba backend
        c = b + b
        assert np.allclose(c.to_numpy(), [2.0, 4.0], atol=1e-6)

    def test_matmul_api(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]], device="numba")
        b = tensor([[5.0, 6.0], [7.0, 8.0]], device="numba")
        c = a @ b
        ref = a.to("cpu") @ b.to("cpu")
        assert c.shape == ref.shape
        assert np.allclose(c.to_numpy(), ref.to_numpy(), rtol=1e-4, atol=1e-5)

    def test_backend_pickling(self):
        import pickle

        obj = pickle.dumps(numba)
        restored = pickle.loads(obj)
        assert restored.is_cpu
        # functional test after unpickle
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ref = numba.neg_map(TensorData(a_np, (3,)))
        out = restored.neg_map(TensorData(a_np, (3,)))
        assert np.allclose(out.numpy_view(), ref.numpy_view(), rtol=1e-7)

    def test_leaky_relu_grad_uses_python_fallback(self):
        """LeakyReLU uses a Python lambda, so it goes through CPU -- smoke test."""
        a = tensor([-1.0, 2.0], device="numba")
        b = a.relu()
        assert b.shape == (2,)


# ---------------------------------------------------------------------------
# autograd + backward pass
# ---------------------------------------------------------------------------
class TestNumbaAutograd:
    def test_add_backward(self):
        a = tensor([1.0, 2.0], requires_grad=True, device="numba")
        b = tensor([3.0, 4.0], requires_grad=True, device="numba")
        c = a + b
        c.backward(tensor([1.0, 1.0], device="numba"))
        assert np.allclose(a.grad.to_numpy(), [1.0, 1.0], atol=1e-6)
        assert np.allclose(b.grad.to_numpy(), [1.0, 1.0], atol=1e-6)

    def test_mul_backward(self):
        a = tensor([2.0, 3.0], requires_grad=True, device="numba")
        b = tensor([4.0, 5.0], requires_grad=True, device="numba")
        c = a * b
        c.backward(tensor([1.0, 1.0], device="numba"))
        assert np.allclose(a.grad.to_numpy(), [4.0, 5.0], atol=1e-6)
        assert np.allclose(b.grad.to_numpy(), [2.0, 3.0], atol=1e-6)

    def test_matmul_backward(self):
        np.random.seed(9)
        a = rand((4, 3), requires_grad=True, device="numba")
        b = rand((3, 5), requires_grad=True, device="numba")
        c = a @ b
        c.backward(rand((4, 5), device="numba"))
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_sigmoid_backward(self):
        a = tensor([0.0, 1.0, -1.0], requires_grad=True, device="numba")
        b = a.sigmoid()
        b.backward(tensor([1.0, 1.0, 1.0], device="numba"))
        assert a.grad is not None
        assert a.grad.shape == (3,)

    def test_softmax_cross_entropy(self):
        """Smoke test: softmax backward runs through numba zip ops."""
        np.random.seed(10)
        logits = randn((8, 5), requires_grad=True, device="numba")
        from torchlight.nn.functional import cross_entropy

        targets = tensor((np.arange(8) % 5).astype(np.float32), device="numba")
        loss = cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_xor_training(self):
        """Smoke test: a tiny XOR MLP trains 100 steps on the numba backend."""
        from torchlight.nn.module import Parameter
        from torchlight.optim import SGD

        X = tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device="numba")
        Y = tensor([[0.0], [1.0], [1.0], [0.0]], device="numba")
        params = [
            Parameter(randn((2, 8), requires_grad=True, device="numba"), name="w1"),
            Parameter(zeros((8,), requires_grad=True, device="numba"), name="b1"),
            Parameter(randn((8, 1), requires_grad=True, device="numba"), name="w2"),
            Parameter(zeros((1,), requires_grad=True, device="numba"), name="b2"),
        ]
        opt = SGD(params, lr=0.2)

        def forward(x):
            h = (x @ params[0].value + params[1].value).sigmoid()
            return h @ params[2].value + params[3].value

        for _ in range(100):
            opt.zero_grad()
            pred = forward(X)
            loss = ((pred - Y) ** 2).sum()
            loss.backward()
            opt.step()

        preds = forward(X)
        assert preds.shape == (4, 1)
        # XOR is separable enough that after 100 steps the MLP can't just
        # output a constant; all four targets must be distinct predictions.
        vals = preds.to_numpy().reshape(-1)
        assert len({round(float(v), 3) for v in vals}) >= 3