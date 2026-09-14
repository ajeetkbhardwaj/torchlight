"""
Loss modules wrapping :mod:`torchlight.nn.functional` (stateless).
"""

from __future__ import annotations

from typing import Any

from ..tensor import Tensor
from . import functional as F
from .module import Module


class _Loss(Module):
    """Base class for losses: no learning occurs, so only ``forward`` matters."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        return self.forward(input, target)

    def extra_repr(self) -> str:
        return ""


class MSELoss(_Loss):
    """Mean squared error: ``mean((input - target)^2)``."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)


class L1Loss(_Loss):
    """Mean absolute error: ``mean(|input - target|)``."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target)


class CrossEntropyLoss(_Loss):
    """
    Cross entropy between logits and class-index targets.

    Args:
        dim: the class dimension in ``input`` (default ``-1``).
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, self.dim)


class BCELoss(_Loss):
    """
    Binary cross entropy (expects ``input`` in ``(0, 1)``).

    Prefer :class:`BCEWithLogitsLoss` for training stability.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target)


class BCEWithLogitsLoss(_Loss):
    """Log-domain-stable binary cross entropy for raw logits."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target)


__all__ = ["MSELoss", "L1Loss", "CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss"]