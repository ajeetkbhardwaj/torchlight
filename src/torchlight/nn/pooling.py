"""
Pooling layers (2D).
"""

from __future__ import annotations

from typing import Tuple, Union

from ..tensor import Tensor
from . import functional as F
from .module import Module


def _to_pair(kernel: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Normalise a single int to a ``(kh, kw)`` pair."""
    return (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)


class _Pool2d(Module):
    """Shared plumbing for 2D pooling layers."""

    def __init__(self, kernel: Union[int, Tuple[int, int]]):
        super().__init__()
        self.kernel = _to_pair(kernel)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel}"


class MaxPool2d(_Pool2d):
    """Max pooling over a ``(B, C, H, W)`` input."""

    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(input, self.kernel)

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel})"


class AvgPool2d(_Pool2d):
    """Average pooling over a ``(B, C, H, W)`` input."""

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(input, self.kernel)

    def __repr__(self) -> str:
        return f"AvgPool2d(kernel_size={self.kernel})"