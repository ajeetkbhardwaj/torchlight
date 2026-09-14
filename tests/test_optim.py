"""Optimizer tests: each optimizer must reduce a quadratic loss."""

import numpy as np

import torchlight as tl

X_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
Y_data = np.array([[5.0], [11.0]], dtype=np.float32)  # perfect y = 3*x1 + x2


class Lin(tl.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = tl.nn.Linear(2, 1, bias=True)

    def forward(self, x):
        return self.w(x)


def run_optimizer(kind, lr, steps=500, wd=0.0):
    model = Lin()
    opt = {
        "sgd": lambda: __import__("torchlight.optim", fromlist=["SGD"]).SGD(model.parameters(), lr=lr, weight_decay=wd),
        "adam": lambda: __import__("torchlight.optim", fromlist=["Adam"]).Adam(model.parameters(), lr=lr),
        "adamw": lambda: __import__("torchlight.optim", fromlist=["AdamW"]).AdamW(model.parameters(), lr=lr, weight_decay=wd),
    }[kind]()
    last = None
    for _ in range(steps):
        opt.zero_grad()
        loss = tl.nn.functional.mse_loss(model(tl.tensor(X_data)), tl.tensor(Y_data))
        loss.backward()
        opt.step()
        last = float(loss.to_numpy().reshape(-1)[0])
    return last


def test_sgd_converges():
    assert run_optimizer("sgd", lr=0.05) < 1e-6


def test_sgd_with_weight_decay_converges():
    assert run_optimizer("sgd", lr=0.05, wd=1e-3) < 1e-4


def test_adam_converges():
    # Adam's per-coordinate scaling can occasionally crawl near the optimum;
    # accept a small residual rather than demanding machine precision.
    assert run_optimizer("adam", lr=0.1) < 5e-3


def test_adamw_converges():
    # Decoupled weight decay biases the solution away from zero loss.
    assert run_optimizer("adamw", lr=0.1, wd=1e-4) < 5e-2


def test_optimizer_steps_leave_leaf_parameters():
    from torchlight.optim import SGD

    model = Lin()
    w = model.w.weight
    opt = SGD(model.parameters(), lr=0.1)
    loss = tl.nn.functional.mse_loss(model(tl.tensor(X_data)), tl.tensor(Y_data))
    loss.backward()
    old = w.value.to_numpy().copy()
    opt.step()
    assert w.value.is_leaf(), "updated parameter must be a detached leaf"
    assert not np.allclose(w.value.to_numpy(), old)