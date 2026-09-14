"""
Convolution layers: ``Conv1d`` and ``Conv2d``.

Thin parameter-owning wrappers around :func:`torchlight.nn.functional.conv1d`
/ :func:`torchlight.nn.functional.conv2d`.  Weights use PyTorch's conv init
(kaiming uniform with ``a = sqrt(5)``, bias uniform in ``+-1/sqrt(fan_in)``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

from ..tensor import Tensor, zeros
from . import functional as F, init as _init
from .module import Module, Parameter


class _ConvNd(Module):
    """Shared init for the conv modules (weight + optional bias)."""

    def _init_weight_bias(self, weight_shape: Tuple[int, ...], bias_shape: Tuple[int, ...], bias: bool) -> None:
        self.weight = Parameter(_init.kaiming_uniform_(zeros(weight_shape), a=math.sqrt(5)), name="weight")
        if bias:
            bound = 1.0 / math.sqrt(weight_shape[1] * _receptive(weight_shape))
            self.bias = Parameter(_init.uniform_(zeros(bias_shape), -bound, bound), name="bias")
        else:
            self.bias = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )


def _receptive(weight_shape: Tuple[int, ...]) -> int:
    """``kernel`` size (product of the trailing kernel dims)."""
    out = 1
    for s in weight_shape[2:]:
        out *= s
    return out


class Conv1d(_ConvNd):
    """
    Applies a convolution (cross-correlation) over a ``(N, C_in, L)`` input.

    Output length is :math:`(L + 2\\cdot\\text{padding} - K) / \\text{stride} + 1`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._init_weight_bias(
            (out_channels, in_channels, kernel_size),
            (out_channels,),
            bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dims != 3:
            raise ValueError(f"Conv1d expects a 3D input (N, C, L), got {input.shape}")
        return F.conv1d(
            input,
            self.weight.value,
            self.bias.value if self.bias is not None else None,
            self.stride,
            self.padding,
        )


class Conv2d(_ConvNd):
    """
    Applies a convolution (cross-correlation) over a ``(N, C_in, H, W)`` input.

    Output shape is
    :math:`(N, C_{out}, (H + 2p_h - K_h)/s_h + 1, (W + 2p_w - K_w)/s_w + 1)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = F._to_int_pair(kernel_size, "kernel_size")
        self.stride = stride
        self.padding = padding

        self._init_weight_bias(
            (out_channels, in_channels, *self.kernel_size),
            (out_channels,),
            bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dims != 4:
            raise ValueError(f"Conv2d expects a 4D input (N, C, H, W), got {input.shape}")
        return F.conv2d(
            input,
            self.weight.value,
            self.bias.value if self.bias is not None else None,
            self.stride,
            self.padding,
        )


__all__ = ["Conv1d", "Conv2d"]