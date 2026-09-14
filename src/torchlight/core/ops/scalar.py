"""
Scalar mathematical operators and their derivatives.

This is the *mathematical vocabulary* of Torchlight.  Every neural-network
building block (activations, reductions, losses) is ultimately defined in
terms of these scalar functions.

Each operator documents three things:

* its definition :math:`f(x)` or :math:`f(x, y)`,
* a numerically stable implementation where relevant,
* a ``*_back`` derivative companion used for gradient checking.

The tensor-level autograd ops in :mod:`torchlight.autograd.functions` mirror
these functions but are vectorised over numpy arrays.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Union

# ---------------------------------------------------------------------------
# Elementary operators
# ---------------------------------------------------------------------------

#: Tiny epsilon added inside ``log`` to keep the domain positive.
EPS = 1e-6


def id(x: float) -> float:
    r"""Identity: :math:`f(x) = x`."""
    return x


def add(x: float, y: float) -> float:
    r"""Addition: :math:`f(x, y) = x + y`."""
    return x + y


def sub(x: float, y: float) -> float:
    r"""Subtraction: :math:`f(x, y) = x - y`."""
    return x - y


def mul(x: float, y: float) -> float:
    r"""Multiplication: :math:`f(x, y) = x \cdot y`."""
    return x * y


def neg(x: float) -> float:
    r"""Negation: :math:`f(x) = -x`."""
    return -x


def inv(x: float) -> float:
    r"""Reciprocal: :math:`f(x) = 1/x`."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"""Derivative of :math:`f(x)=1/x`: returns :math:`d \cdot f'(x) = -d / x^2`."""
    return -(1.0 / (x * x)) * d


def pow(x: float, y: float) -> float:
    r"""Power: :math:`f(x, y) = x^y`."""
    return x ** y


def sqrt(x: float) -> float:
    r"""Square root: :math:`f(x) = \sqrt{x}`."""
    return math.sqrt(x)


def log(x: float) -> float:
    r"""Natural log with a small shift: :math:`f(x) = \log(x + \epsilon)`."""
    return math.log(x + EPS)


def log_back(x: float, d: float) -> float:
    r"""Derivative of :math:`f(x)=\log(x+\epsilon)`: :math:`d / (x+\epsilon)`."""
    return d / (x + EPS)


def exp(x: float) -> float:
    r"""Exponential: :math:`f(x) = e^x`."""
    return math.exp(x)


# ---------------------------------------------------------------------------
# Nonlinearities (used as activations)
# ---------------------------------------------------------------------------
def sigmoid(x: float) -> float:
    r"""
    Sigmoid: :math:`f(x) = 1 / (1 + e^{-x})`.

    Implemented piecewise for numerical stability: for :math:`x < 0` we use
    the equivalent :math:`e^x / (1 + e^x)` to avoid overflow of :math:`e^{-x}`.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def tanh(x: float) -> float:
    r"""Hyperbolic tangent: :math:`f(x) = \tanh(x)`."""
    return math.tanh(x)


def relu(x: float) -> float:
    r"""Rectified linear unit: :math:`f(x) = \max(x, 0)`."""
    return x if x > 0 else 0.0


def relu_back(x: float, d: float) -> float:
    r"""Derivative of ReLU: returns :math:`d \cdot \mathbb{1}[x > 0]`."""
    return d if x > 0 else 0.0


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    r"""Leaky ReLU: :math:`f(x) = x` if :math:`x > 0` else :math:`\alpha x`."""
    return x if x > 0 else alpha * x


def leaky_relu_back(x: float, d: float, alpha: float = 0.01) -> float:
    r"""Derivative of leaky ReLU: :math:`d` if :math:`x>0` else :math:`\alpha d`."""
    return d if x > 0 else alpha * d


def sqrt_back(x: float, d: float) -> float:
    r"""Derivative of :math:`f(x)=\sqrt{x}`: :math:`d / (2\sqrt{x})` (guarded).

    Returns ``0`` where ``x <= 0`` so negative-domain inputs don't produce NaNs.
    """
    return d / (2.0 * sqrt(x)) if x > 0 else 0.0


def abs_back(x: float, d: float) -> float:
    r"""Derivative of :math:`f(x)=|x|`: :math:`\mathrm{sign}(x) \cdot d`.

    The subgradient convention at ``x == 0`` is ``0``.
    """
    return d if x > 0 else (-d if x < 0 else 0.0)


# ---------------------------------------------------------------------------
# Comparisons (return 1.0 / 0.0 as floats so they compose with autograd)
# ---------------------------------------------------------------------------
def lt(x: float, y: float) -> float:
    r""":math:`f(x,y) = 1.0` if :math:`x < y` else :math:`0.0`."""
    return 1.0 if x < y else 0.0


def gt(x: float, y: float) -> float:
    r""":math:`f(x,y) = 1.0` if :math:`x > y` else :math:`0.0`."""
    return 1.0 if x > y else 0.0


def eq(x: float, y: float) -> float:
    r""":math:`f(x,y) = 1.0` if :math:`x == y` else :math:`0.0`."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    r""":math:`f(x,y) = \max(x, y)`."""
    return x if x > y else y


def min(x: float, y: float) -> float:
    r""":math:`f(x,y) = \min(x, y)`."""
    return x if x < y else y


def is_close(x: float, y: float) -> float:
    r""":math:`f(x,y) = 1.0` if :math:`|x - y| < 10^{-2}` else :math:`0.0`."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


# ---------------------------------------------------------------------------
# Higher-order helpers: map / zip / reduce
# ---------------------------------------------------------------------------
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], List[float]]:
    """
    Higher-order ``map``: apply ``fn`` to every element of a list.

        >>> double = map(lambda x: x * 2)
        >>> double([1, 2, 3])
        [2, 4, 6]
    """
    return lambda ls: [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], List[float]]:
    """
    Higher-order ``zipWith``: combine two lists elementwise with ``fn``.

        >>> add_lists = zipWith(add)
        >>> add_lists([1, 2], [10, 20])
        [11, 22]
    """
    return lambda ls1, ls2: [fn(a, b) for a, b in zip(ls1, ls2)]


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """
    Higher-order ``reduce``: fold ``fn`` over a list starting from ``start``.

        >>> total = reduce(add, 0.0)
        >>> total([1, 2, 3, 4])
        10.0
    """
    return lambda ls: _fold(fn, start, ls)


def _fold(fn: Callable[[float, float], float], start: float, ls: Iterable[float]) -> float:
    acc = start
    for x in ls:
        acc = fn(acc, x)
    return acc


def sum(ls: Iterable[float]) -> float:
    """Sum a list with ``reduce(add, 0.0)``."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Multiply a list together with ``reduce(mul, 1.0)``."""
    return reduce(mul, 1.0)(ls)


__all__ = [
    "EPS",
    "id",
    "add",
    "sub",
    "mul",
    "neg",
    "inv",
    "inv_back",
    "pow",
    "sqrt",
    "log",
    "log_back",
    "exp",
    "sigmoid",
    "tanh",
    "relu",
    "relu_back",
    "leaky_relu",
    "leaky_relu_back",
    "sqrt_back",
    "abs_back",
"lt",
    "gt",
    "max",
    "min",
    "is_close",
    "map",
    "zipWith",
    "reduce",
    "sum",
    "prod",
]