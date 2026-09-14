"""
Adam and AdamW optimizers (adaptive moment estimation).

Adam keeps per-parameter first and second moment estimates and applies
bias correction::

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g^2
    m_hat = m / (1 - beta1^t)
    v_hat = v / (1 - beta2^t)
    w    -= lr * m_hat / (sqrt(v_hat) + eps)

``AdamW`` applies *decoupled* L2 weight decay directly to the weights
(``w *= (1 - lr * lambda)``) instead of adding it to the gradient.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from ..nn.module import Parameter
from ..tensor import Tensor
from .optimizer import Optimizer


class Adam(Optimizer):
    """Adam optimizer with bias-corrected first/second moments."""

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(parameters, defaults=dict(lr=lr, betas=betas, eps=eps))
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.parameters:
            grad = self._grad(p)
            if grad is None:
                continue
            state = self._param_state(p)

            # L2 (coupled) weight decay.
            g = grad.detach()
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.value

            # Lazy moment-buffer initialisation.
            if "exp_avg" not in state or state["exp_avg"] is None:
                state["exp_avg"] = g.zeros(g.shape)
                state["exp_avg_sq"] = g.zeros(g.shape)
                state["step"] = 0

            state["step"] += 1
            t = state["step"]

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            exp_avg = self.beta1 * exp_avg + (1.0 - self.beta1) * g
            exp_avg_sq = self.beta2 * exp_avg_sq + (1.0 - self.beta2) * (g * g)

            # Bias correction (python float powers: cheap and exact).
            bias1 = 1.0 - self.beta1 ** t
            bias2 = 1.0 - self.beta2 ** t
            m_hat = exp_avg / bias1
            v_hat = exp_avg_sq / bias2
            denom = v_hat.sqrt() + self.eps

            state["exp_avg"] = exp_avg.detach()
            state["exp_avg_sq"] = exp_avg_sq.detach()

            p.update(p.value - self.lr * (m_hat / denom))

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )


class AdamW(Optimizer):
    """
    Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019):

    ``w <- w * (1 - lr * lambda) - lr * m_hat / (sqrt(v_hat) + eps)``

    This behaves differently from ``Adam(weight_decay=...)`` for large
    weights -- recommend using this variant with modern transformers.
    """

    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(parameters, defaults=dict(lr=lr, betas=betas, eps=eps))
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def step(self) -> None:
        for p in self.parameters:
            grad = self._grad(p)
            if grad is None:
                continue
            state = self._param_state(p)

            g = grad.detach()

            if "exp_avg" not in state or state["exp_avg"] is None:
                state["exp_avg"] = g.zeros(g.shape)
                state["exp_avg_sq"] = g.zeros(g.shape)
                state["step"] = 0

            state["step"] += 1
            t = state["step"]

            exp_avg = self.beta1 * state["exp_avg"] + (1.0 - self.beta1) * g
            exp_avg_sq = self.beta2 * state["exp_avg_sq"] + (1.0 - self.beta2) * (g * g)

            bias1 = 1.0 - self.beta1 ** t
            bias2 = 1.0 - self.beta2 ** t
            m_hat = exp_avg / bias1
            v_hat = exp_avg_sq / bias2
            denom = v_hat.sqrt() + self.eps

            state["exp_avg"] = exp_avg.detach()
            state["exp_avg_sq"] = exp_avg_sq.detach()

            # Decoupled weight decay: shrink the weight directly.
            decayed = p.value * (1.0 - self.lr * self.weight_decay)
            p.update(decayed - self.lr * (m_hat / denom))

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay})"
        )