"""
Stochastic Gradient Descent with optional momentum / weight decay.
"""

from __future__ import annotations

from typing import Iterable

from ..nn.module import Parameter
from ..tensor import Tensor
from .optimizer import Optimizer


class SGD(Optimizer):
    r"""
    SGD update:

    .. math::
        w_{t+1} = w_t - \eta \cdot \hat{g}_t

    with optional L2 weight decay (:math:`\hat g = g + \lambda w`) and
    momentum (:math:`v = \mu v_{t-1} + \hat g`, `nesterov` variant uses
    :math:`\hat g + \mu v`).
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if momentum < 0.0:
            raise ValueError("momentum must be >= 0")
        if nesterov and momentum == 0.0:
            raise ValueError("nesterov requires momentum > 0")
        super().__init__(parameters, defaults=dict(lr=lr))
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    def step(self) -> None:
        for p in self.parameters:
            grad = self._grad(p)
            if grad is None:
                continue

            base = p.value
            g = grad.detach()

            # L2 weight decay: add lambda * w to the gradient.
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * base

            if self.momentum != 0.0:
                state = self._param_state(p)
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = g.zeros(g.shape)
                    state["momentum_buffer"] = buf
                if self.dampening != 0.0:
                    buf = self.momentum * buf + (1.0 - self.dampening) * g
                else:
                    buf = self.momentum * buf + g
                state["momentum_buffer"] = buf.detach()
                g = buf + self.momentum * buf if self.nesterov else buf

            p.update(base - self.lr * g)

    def __repr__(self) -> str:
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, nesterov={self.nesterov})"
        )