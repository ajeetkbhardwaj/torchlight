"""
Scalar-level differentiable operations (``ScalarFunction`` subclasses).

Mirrors :mod:`torchlight.autograd.functions` but over plain Python ``float``
values instead of tensors: each subclass defines ``forward(ctx, *floats)`` and
``backward(ctx, d_output)``, and :meth:`ScalarFunction.apply` wraps every input
in a :class:`~torchlight.scalar.scalar.Scalar` record so reverse-mode autodiff
works exactly like the tensor engine.

The mathematical definitions reuse :mod:`torchlight.core.ops` (the scalar
spec), keeping the tensor and scalar versions of every operator in sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from ..autograd.autodiff import Context
from ..core import ops as scalar_ops

if TYPE_CHECKING:
    from .scalar import Scalar


def _wrap_tuple(x: Any) -> Tuple[Any, ...]:
    """Turn a scalar / single object into a tuple (so backward needn't)."""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    :class:`Scalar` variables (static, never instantiated).

    ``apply`` is the only public entry point: it coerces every input to a
    ``Scalar``, runs the numeric ``forward``, and records a
    ``ScalarHistory`` so ``backward()`` can push gradients back.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return _wrap_tuple(cls.backward(ctx, d_out))  # type: ignore[attr-defined]

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore[attr-defined]

    @classmethod
    def apply(cls, *vals: Any) -> "Scalar":
        # Deferred import breaks the scalar <-> scalar_functions cycle.
        from .scalar import Scalar, ScalarHistory

        scalars: list = []
        raw_vals: list = []
        for v in vals:
            if isinstance(v, Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                s = Scalar(v)  # wrap the constant; it becomes a (tracked) leaf
                scalars.append(s)
                raw_vals.append(s.data)

        ctx = Context(no_grad=False)
        c = cls._forward(ctx, *raw_vals)
        back = ScalarHistory(cls, ctx, tuple(scalars))
        return Scalar(c, back)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
class Add(ScalarFunction):
    r"""Addition :math:`f(x, y) = x + y`."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return d_output, d_output


class Mul(ScalarFunction):
    r"""Multiplication :math:`f(x, y) = x \cdot y`."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Neg(ScalarFunction):
    r"""Negation :math:`f(x) = -x`."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -d_output


class Inv(ScalarFunction):
    r"""Reciprocal :math:`f(x) = 1/x`."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return scalar_ops.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return scalar_ops.inv_back(a, d_output)


class Exp(ScalarFunction):
    r"""Exponential :math:`f(x) = e^x`."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        out = scalar_ops.exp(a)
        ctx.save_for_backward(out)  # dz/dx = e^x = out
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (out,) = ctx.saved_values
        return d_output * out


class Log(ScalarFunction):
    r"""Natural log :math:`f(x) = \log(x + 10^{-6})` (shifted, as the spec)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return scalar_ops.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return scalar_ops.log_back(a, d_output)


# ---------------------------------------------------------------------------
# Nonlinearities
# ---------------------------------------------------------------------------
class Sigmoid(ScalarFunction):
    r"""Logistic sigmoid :math:`f(x) = 1/(1 + e^{-x})` (stable)."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        out = scalar_ops.sigmoid(a)
        ctx.save_for_backward(out)  # dz/dx = sigma (1 - sigma)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (sigma,) = ctx.saved_values
        return sigma * (1.0 - sigma) * d_output


class Tanh(ScalarFunction):
    r"""Hyperbolic tangent :math:`f(x) = \tanh(x)`."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return scalar_ops.tanh(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        t = scalar_ops.tanh(a)
        return (1.0 - t * t) * d_output


class ReLU(ScalarFunction):
    r"""Rectified linear unit :math:`f(x) = \max(x, 0)`."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return scalar_ops.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return scalar_ops.relu_back(a, d_output)


# ---------------------------------------------------------------------------
# Comparisons (non-differentiable: zero gradients)
# ---------------------------------------------------------------------------
class LT(ScalarFunction):
    r""":math:`f(x, y) = 1.0` if :math:`x < y` else :math:`0.0`."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0


class EQ(ScalarFunction):
    r""":math:`f(x, y) = 1.0` if :math:`x == y` else :math:`0.0`."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0


__all__ = [
    "ScalarFunction",
    "Add",
    "Mul",
    "Neg",
    "Inv",
    "Exp",
    "Log",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "LT",
    "EQ",
]