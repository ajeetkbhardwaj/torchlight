"""
``Dropout``: regularisation layer.

Active only while the module is in *training* mode (``module.train()``).
In evaluation mode it is the identity, matching PyTorch's behaviour.
"""

from __future__ import annotations

from ..tensor import Tensor
from . import functional as F
from .module import Module


class Dropout(Module):
    """Randomly zero elements with probability ``p`` (inverted dropout)."""

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        super().__init__()
        #: Dropout probability.
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, training=self.training)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"