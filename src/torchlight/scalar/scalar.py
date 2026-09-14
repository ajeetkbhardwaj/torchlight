"""
``Scalar`` autodiff node -- a single float that participates in a computation
graph for reverse-mode automatic differentiation.

Mirrors ``minitorch.Scalar``: every :class:`ScalarFunction` subclass wraps its
inputs via :meth:`ScalarFunction.apply`, building ``ScalarHistory`` records so
:meth:`Scalar.backward` can push gradients to every tracked leaf.

The recommended usage pattern is::

    from torchlight.scalar import Scalar, derivative_check

    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x * y + x.sigmoid()
    z.backward()
    assert x.derivative is not None
    derivative_check(lambda x, y: x * y + x.sigmoid(), x, y)

The scalar ``backpropagate`` traversal is intentionally kept separate from the
tensor-level one in :mod:`torchlight.autograd.autodiff` so gradients stay as
plain Python ``float`` values (no ``_raw_add`` or tensor broadcast needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ..autograd.autodiff import Context, central_difference, topological_sort

ScalarLike = Union[float, int, "Scalar"]


# ---------------------------------------------------------------------------
# History: one edge of the scalar compute graph
# ---------------------------------------------------------------------------
@dataclass
class ScalarHistory:
    """
    Records how a scalar was produced during a forward pass.

    A leaf (user-created) scalar has ``last_fn is None``.  An intermediate
    scalar produced by a ``ScalarFunction`` has the function, its saved context,
    and the parent ``Scalar`` nodes.
    """

    last_fn: Optional[Type["ScalarFunction"]] = None  # type: ignore[forward-ref]
    ctx: Optional[Context] = None
    inputs: Sequence["Scalar"] = ()


# ---------------------------------------------------------------------------
# Scalar variable
# ---------------------------------------------------------------------------
_var_count = 0


class Scalar:
    """
    A Python float that tracks its computation history.

    By default, a ``Scalar`` is a *tracked leaf* (its ``history`` is an
    empty ``ScalarHistory``), so gradients accumulate directly into it.  Pass
    ``back=None`` explicitly, or call ``requires_grad_(False)``, to turn it
    into a non-differentiable constant.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: Optional[ScalarHistory] = ScalarHistory(),
        name: Optional[str] = None,
    ) -> None:
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        self.name = name if name is not None else str(self.unique_id)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __str__(self) -> str:
        return repr(self)

    # ------------------------------------------------------------------
    # Autograd interface
    # ------------------------------------------------------------------
    def requires_grad(self) -> bool:
        """``True`` when this node is being tracked (history not None)."""
        return self.history is not None

    def requires_grad_(self, flag: bool = True) -> "Scalar":
        """Toggle tracking in place; returns ``self``."""
        if flag and self.history is None:
            self.history = ScalarHistory()
        elif not flag:
            self.history = None
        return self

    def is_leaf(self) -> bool:
        """True when this variable was created by the user (empty history)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True when ``history`` is ``None`` (no tracking)."""
        return self.history is None

    @property
    def parents(self) -> Iterable["Scalar"]:
        assert self.history is not None
        return self.history.inputs

    def accumulate_derivative(self, x: float) -> None:
        """Add ``x`` to this leaf's accumulated gradient."""
        assert self.is_leaf(), "Only leaf variables can accumulate derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Scalar", Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        grads = h.last_fn._backward(h.ctx, d_output)  # type: ignore[attr-defined]
        return list(zip(h.inputs, grads))

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Run reverse-mode autodiff from this scalar's graph root.

        Args:
            d_output: gradient seed (defaults to ``1.0``).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __mul__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import Mul
        return Mul.apply(self, b)

    def __rmul__(self, b: ScalarLike) -> "Scalar":
        return self * b

    def __truediv__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import Inv, Mul
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import Inv, Mul
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import Add
        return Add.apply(self, b)

    def __radd__(self, b: ScalarLike) -> "Scalar":
        return self + b

    def __sub__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import Add, Neg
        return Add.apply(self, Neg.apply(b))

    def __rsub__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import Add, Neg
        return Add.apply(Neg.apply(self), b)

    def __neg__(self) -> "Scalar":
        from .scalar_functions import Neg
        return Neg.apply(self)

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------
    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import LT
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> "Scalar":
        from .scalar_functions import LT
        return LT.apply(b, self)

    def __eq__(self, b: object) -> "Scalar":  # type: ignore[override]
        from .scalar_functions import EQ
        if not isinstance(b, (int, float, Scalar)):
            return NotImplemented
        return EQ.apply(self, b)

    # ------------------------------------------------------------------
    # Common functionals
    # ------------------------------------------------------------------
    def log(self) -> "Scalar":
        from .scalar_functions import Log
        return Log.apply(self)

    def exp(self) -> "Scalar":
        from .scalar_functions import Exp
        return Exp.apply(self)

    def sigmoid(self) -> "Scalar":
        from .scalar_functions import Sigmoid
        return Sigmoid.apply(self)

    def tanh(self) -> "Scalar":
        from .scalar_functions import Tanh
        return Tanh.apply(self)

    def relu(self) -> "Scalar":
        from .scalar_functions import ReLU
        return ReLU.apply(self)


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------
def backpropagate(variable: Scalar, deriv: float = 1.0) -> None:
    """
    Reverse-mode differentiation over the scalar graph rooted at ``variable``.

    Walks from root toward leaves, accumulating ``deriv`` into every
    tracked, non-constant leaf via :meth:`Scalar.accumulate_derivative`.
    """
    if variable.is_constant():
        return

    order = topological_sort(variable)  # leaves -> root
    node_grads: dict[int, Any] = {variable.unique_id: deriv}

    for node in reversed(order):
        if node.unique_id not in node_grads:
            continue
        d_output = node_grads.pop(node.unique_id)

        if node.is_leaf():
            node.accumulate_derivative(d_output)
            continue

        for parent, d_parent in node.chain_rule(d_output):
            if not isinstance(parent, Scalar) or parent.is_constant():
                continue
            cur = node_grads.get(parent.unique_id)
            if cur is None:
                node_grads[parent.unique_id] = d_parent
            else:
                node_grads[parent.unique_id] = cur + d_parent


# ---------------------------------------------------------------------------
# Gradient checking
# ---------------------------------------------------------------------------
def derivative_check(
    f: Callable[..., "Scalar"],
    *scalars: "Scalar",
) -> None:
    """
    Check that the analytic derivatives stored in ``scalars`` after calling
    ``f(*scalars).backward()`` match a central-difference estimate.

    Uses :func:`torchlight.autograd.autodiff.central_difference`, which
    perturbs each input via ``+= epsilon`` (using the arithmetic operators of
    the ``Scalar`` wrapper).

    Raises ``AssertionError`` if the analytic and numeric values disagree
    beyond ``atol=rtol=1e-2``.
    """
    out = f(*scalars)
    out.backward()

    err_msg = (
        "Derivative check at arguments f(%s) and received derivative "
        "f'=%f for argument %d, but was expecting derivative f'=%f from "
        "central difference."
    )
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        check_val = check.data if isinstance(check, Scalar) else float(check)
        assert x.derivative is not None, (
            f"variable {x} has no accumulated derivative after backward()"
        )
        np.testing.assert_allclose(
            x.derivative,
            check_val,
            rtol=1e-2,
            atol=1e-2,
            err_msg=err_msg
            % (str([s.data for s in scalars]), x.derivative, i, check_val),
        )


__all__ = [
    "Scalar",
    "ScalarHistory",
    "ScalarLike",
    "backpropagate",
    "derivative_check",
]