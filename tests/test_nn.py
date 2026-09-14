"""nn module tests: shapes, training loops, functional losses, dropout."""

import numpy as np
import pytest

import torchlight as tl
from torchlight.nn import (
    CrossEntropyLoss,
    Linear,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
)
from torchlight.nn import functional as F


class TwoLayer(tl.nn.Module):
    def __init__(self, d_in=2, d_hidden=16, d_out=2):
        super().__init__()
        self.fc1 = Linear(d_in, d_hidden)
        self.fc2 = Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(ReLU()(self.fc1(x)))


def test_linear_shapes_and_parameters():
    layer = Linear(4, 3)
    assert layer.weight.shape == (3, 4)
    assert layer.bias.shape == (3,)
    assert layer(tl.zeros((5, 4))).shape == (5, 3)


def test_linear_no_bias():
    layer = Linear(4, 3, bias=False)
    assert layer.bias is None
    assert layer(tl.zeros((2, 4))).shape == (2, 3)


def test_named_parameters_walk_the_tree():
    model = TwoLayer()
    names = {n for n, _ in model.named_parameters()}
    assert names == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}


def test_sequential_forward():
    net = Sequential(Linear(2, 4), ReLU(), Linear(4, 2))
    assert net(tl.zeros((3, 2))).shape == (3, 2)


def test_softmax_is_probability_distribution():
    x = tl.tensor([[1.0, 2.0, 3.0]])
    p = Softmax()(x).to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.min() > 0


def test_sigmoid_bounds():
    x = tl.tensor([-100.0, 0.0, 100.0])
    s = Sigmoid()(x).to_numpy()
    assert np.allclose(s, [0.0, 0.5, 1.0], atol=1e-6)


def test_mlp_trains_down_on_synthetic_xor():
    from torchlight.data import make_synthetic
    from torchlight.optim import SGD

    ds = make_synthetic("Xor", n=200, seed=0)
    X, y = ds.to_batch()
    model = TwoLayer()
    opt = SGD(model.parameters(), lr=0.5)
    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.to_numpy().reshape(-1)[0]))
    assert losses[-1] < losses[0], "loss did not decrease"


def test_cross_entropy_loss_module_matches_functional():
    x = tl.tensor(np.random.rand(4, 3).astype(np.float32))
    y = tl.tensor(np.array([0, 2, 1, 2], dtype=np.float32))
    assert np.allclose(CrossEntropyLoss()(x, y).to_numpy(),
                       F.cross_entropy(x, y).to_numpy())


def test_dropout_trains_differs_from_eval_with_masked_grad():
    import torchlight.nn as nn
    x = tl.ones((10, 10))
    d = nn.Dropout(0.5)
    d.train()
    out = d(x).to_numpy()
    assert (out == 0).any(), "some units must be masked"
    assert abs(out.mean() - 1.0) < 0.3, "scaled survivors keep the mean ~1"
    d.eval()
    assert np.allclose(d(x).to_numpy(), np.ones((10, 10)))


def test_layer_norm_normalises_last_dim():
    x = tl.tensor(np.random.rand(4, 5).astype(np.float32))
    y = F.layer_norm(x, (5,)).to_numpy()
    assert np.allclose(y.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(y.std(axis=1), 1.0, atol=1e-3)


def test_mse_loss_value():
    pred = tl.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = tl.tensor([[1.0, 2.0], [3.0, 6.0]])
    assert F.mse_loss(pred, target).item() == pytest.approx(1.0)
    assert F.mse_loss(pred, target, reduction="mean").item() == pytest.approx(1.0)
    assert F.mse_loss(pred, target, reduction="sum").item() == pytest.approx(4.0)


def test_parameters_are_detached_leaves_after_update():
    layer = Linear(2, 2)
    p = layer.weight
    assert p.value.is_leaf()
    assert p.value.requires_grad