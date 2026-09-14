"""
Activation modules: thin wrappers around :mod:`torchlight.nn.functional`.
"""

from __future__ import annotations

from ..tensor import Tensor
from . import functional as F
from .module import Module


class _Activation(Module):
    """Base class: activations have no parameters or child modules."""

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ReLU(_Activation):
    """Rectified linear unit."""

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


class LeakyReLU(_Activation):
    """Leaky ReLU with the given negative slope."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope)

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class Sigmoid(_Activation):
    """Logistic sigmoid."""

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(_Activation):
    """Hyperbolic tangent."""

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class Softmax(_Activation):
    """Softmax over ``dim``."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class LogSoftmax(_Activation):
    """Log-softmax over ``dim`` (log-domain stable)."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, self.dim)

    def __repr__(self) -> str:
        return f"LogSoftmax(dim={self.dim})"


class GELU(_Activation):
    r"""
    Gaussian error linear unit (``tanh`` approximation):

    .. math:: \mathrm{GELU}(x) =
        0.5 x \left(1 + \tanh\left(\sqrt{\tfrac{2}{\pi}}(x + 0.044715 x^3)\right)\right)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

    def __repr__(self) -> str:
        return "GELU()"