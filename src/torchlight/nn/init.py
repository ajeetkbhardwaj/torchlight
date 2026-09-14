"""
Weight initializers, applied to :class:`Parameter` / :class:`Tensor` in place.

Each initializer fills a tensor's storage using the tensor's own in-place
routines (:meth:`Tensor.uniform_`, :meth:`Tensor.normal_`), so the operation
is cheap and never touches the autograd graph.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from ..tensor import Tensor


def _check_tensor(t: Tensor) -> Tensor:
    if not isinstance(t, Tensor):
        raise TypeError(f"Initializer expects a Tensor, got {type(t)!r}")
    return t


def uniform_(t: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """Fill ``t`` in place with ``U(low, high)``."""
    _check_tensor(t).uniform_(low, high)
    return t


def normal_(t: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill ``t`` in place with ``N(mean, std)``."""
    _check_tensor(t).normal_(mean, std)
    return t


def zeros_(t: Tensor) -> Tensor:
    """Fill ``t`` in place with zeros."""
    _check_tensor(t).zeros_()
    return t


def ones_(t: Tensor) -> Tensor:
    """Fill ``t`` in place with ones."""
    _check_tensor(t).fill_(1.0)
    return t


def xavier_uniform_(t: Tensor, gain: float = 1.0) -> Tensor:
    """
    Xavier (Glorot) uniform init.

    ``U(-a, a)`` with ``a = gain * sqrt(6 / (fan_in + fan_out))``.
    """
    t = _check_tensor(t)
    if t.dims < 2:
        raise ValueError(f"xavier_uniform_ needs a >= 2D tensor, got {t.shape}")
    fan_in, fan_out = _fans(t.shape)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(t, -a, a)


def xavier_normal_(t: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier (Glorot) normal init: standard deviation ``gain * sqrt(2/(fan_in+fan_out))``."""
    t = _check_tensor(t)
    if t.dims < 2:
        raise ValueError(f"xavier_normal_ needs a >= 2D tensor, got {t.shape}")
    fan_in, fan_out = _fans(t.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(t, 0.0, std)


def kaiming_uniform_(t: Tensor, a: float = 0.0) -> Tensor:
    """
    Kaiming (He) uniform init.

    ``U(-b, b)`` with ``b = sqrt(6 / (fan_in * (1 + a^2)))`` (defaults to He).
    """
    t = _check_tensor(t)
    if t.dims == 0:
        raise ValueError("kaiming_uniform_ needs at least 1D")
    fan_in, _ = _fans(t.shape)
    gain = math.sqrt(2.0 / (1.0 + a * a))
    bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
    return uniform_(t, -bound, bound)


def kaiming_normal_(t: Tensor, a: float = 0.0) -> Tensor:
    """Kaiming (He) normal init: ``N(0, sqrt(2/(fan_in*(1+a^2))))``."""
    t = _check_tensor(t)
    if t.dims == 0:
        raise ValueError("kaiming_normal_ needs at least 1D")
    fan_in, _ = _fans(t.shape)
    gain = math.sqrt(2.0 / (1.0 + a * a))
    std = gain / math.sqrt(fan_in)
    return normal_(t, 0.0, std)


def _fans(shape: Sequence[int]) -> Tuple[int, int]:
    """Return ``(fan_in, fan_out)`` for a weight-shaped ``shape``."""
    if len(shape) < 2:
        raise ValueError(f"Need at least 2 dimensions for fan computation, got {shape}")
    if len(shape) == 2:
        fan_in, fan_out = shape
    else:
        # Convolution-style weight: out_channels x in_channels x *kernel.
        receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
        fan_in = shape[1] * receptive
        fan_out = shape[0] * receptive
    return fan_in, fan_out


__all__ = [
    "uniform_",
    "normal_",
    "zeros_",
    "ones_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
]