"""
Core autodifferentiation machinery.

This module defines the three small types that make up a computation graph
and the two algorithms that traverse it:

* :class:`Context`  -- scratchpad a function uses to save values for backward.
* :class:`History`  -- records *which* function built a tensor, from what.
* :class:`Variable` -- protocol for anything with a ``unique_id`` / parents.

* :func:`topological_sort` -- order the graph so children precede parents.
* :func:`backpropagate`    -- the reverse-mode (backprop) traversal.

The reference implementation this is influenced by is the MiniTorch autodiff
layer (Cornell Tech), cleaned up and fully documented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tensor.tensor import Tensor
    from .functions import Function


# ---------------------------------------------------------------------------
# Gradient checking helper
# ---------------------------------------------------------------------------
def central_difference(
    f: Callable[..., float], *vals: float, arg: int = 0, epsilon: float = 1e-6
) -> float:
    r"""
    Central-difference approximation of a scalar function's partial
    derivative with respect to ``vals[arg]``:

    .. math:: f'(x) \approx \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}

    Used by the test-suite to verify each autograd op against its numeric
    derivative.
    """
    vals1 = list(vals)
    vals2 = list(vals)
    vals1[arg] += epsilon
    vals2[arg] -= epsilon
    return (f(*vals1) - f(*vals2)) / (2 * epsilon)


# ---------------------------------------------------------------------------
# The graph node protocol
# ---------------------------------------------------------------------------
class Variable:
    """
    Minimum contract every graph node (a :class:`Tensor`) must fulfil so the
    algorithms in this module stay decoupled from the concrete Tensor class.
    """

    unique_id: int

    def accumulate_derivative(self, x: Any) -> None: ...

    @property
    def parents(self) -> Iterable["Variable"]: ...

    def is_leaf(self) -> bool: ...
    def is_constant(self) -> bool: ...
    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]: ...


# ---------------------------------------------------------------------------
# Function context
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """
    Scratchpad handed from a :class:`Function`'s ``forward`` to its
    ``backward``.

    A forward may call :meth:`save_for_backward` to stash any values the
    backward pass needs (e.g. the inputs), because the backward pass only
    receives the output gradient.
    """

    #: If ``True`` the forward was run in no-grad mode (no inputs need grad).
    no_grad: bool = False

    #: Values saved by ``save_for_backward`` during forward.
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store ``values`` so ``backward`` can access them later."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Alias matching PyTorch's ``ctx.saved_tensors`` naming."""
        return self.saved_values


# ---------------------------------------------------------------------------
# History: one edge of the compute graph
# ---------------------------------------------------------------------------
@dataclass
class History:
    """
    Records how a tensor was produced during a forward pass.

    A leaf tensor (created by the user) has ``last_fn is None``.  An
    intermediate tensor has a non-``None`` ``last_fn``, the ``ctx`` from its
    forward, and the ``inputs`` (parents) it was built from.

    Attributes:
        last_fn: the :class:`Function` class that produced this tensor.
        ctx: the :class:`Context` that function used (saved values).
        inputs: the parent tensors this tensor was computed from.
    """

    last_fn: Optional[Type["Function"]] = None
    ctx: Optional[Context] = None
    inputs: Tuple["Tensor", ...] = field(default_factory=tuple)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        fn_name = self.last_fn.__name__ if self.last_fn else "<leaf>"
        return f"History({fn_name}, n_inputs={len(self.inputs)})"


# ---------------------------------------------------------------------------
# JIT recording hook (used by torchlight.jit.trace)
# ---------------------------------------------------------------------------
#: The autograd layer is the natural choke point through which *every*
#: tensor op passes, so the trace recorder registers a callback here.  The
#: ``jit`` package sets it while tracing; otherwise it stays ``None``.
_op_callback: Optional[Callable[[Type["Function"], Tuple["Tensor", ...], "Tensor"], None]] = None


def set_op_callback(cb: Optional[Callable[..., None]]) -> None:
    """Install/clear the global op-recording callback."""
    global _op_callback
    _op_callback = cb


def _maybe_record(
    fn: Type["Function"], inputs: Tuple["Tensor", ...], output: "Tensor"
) -> None:
    """Fire any installed op callback (no-op in normal execution)."""
    if _op_callback is not None:
        _op_callback(fn, inputs, output)


# ---------------------------------------------------------------------------
# Graph traversal algorithms
# ---------------------------------------------------------------------------
def topological_sort(variable: Variable) -> List[Variable]:
    """
    Return every non-constant node in the graph ending at ``variable`` in
    topological order *from the leaves to the root*.

    Guarantees: for every edge ``parent -> child``, ``parent`` appears before
    ``child``, so reverse iteration in :func:`backpropagate` delivers
    parents-before-children processing.
    """
    visited: set[int] = set()
    order: List[Variable] = []

    def visit(v: Variable) -> None:
        if v.unique_id in visited:
            return
        visited.add(v.unique_id)
        if not v.is_constant() and not v.is_leaf():
            for parent in v.parents:
                visit(parent)
        order.append(v)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Reverse-mode differentiation over the dynamic graph rooted at
    ``variable``.

    Walks the graph from ``variable`` back to the leaves, accumulating a
    gradient into every non-constant leaf through
    :meth:`Variable.accumulate_derivative`.

    Args:
        variable: the tensor whose backward was called (the graph root).
        deriv: the gradient of the loss with respect to ``variable``.
    """
    # If the root is a plain constant there is nothing to do.
    if variable.is_constant():
        return

    order = topological_sort(variable)  # leaves -> root
    node_grads: Dict[int, Any] = {variable.unique_id: deriv}

    # Walk from the root back towards the leaves.
    for node in reversed(order):
        if node.unique_id not in node_grads:
            continue  # no gradient reached this node (dead branch)
        d_output = node_grads.pop(node.unique_id)

        if node.is_leaf():
            # Seeds must be provided by the user.  If not, treat as constant.
            node.accumulate_derivative(d_output)
            continue

        # Keep only non-constant parents -- constants never take gradients.
        for parent, d_parent in node.chain_rule(d_output):
            if parent.is_constant():
                continue
            cur = node_grads.get(parent.unique_id)
            if cur is None:
                node_grads[parent.unique_id] = d_parent
            else:
                # Multiple children feeding the same parent: gradient sums.
                node_grads[parent.unique_id] = cur._raw_add(d_parent)


__all__ = [
    "central_difference",
    "Variable",
    "Context",
    "History",
    "topological_sort",
    "backpropagate",
    "set_op_callback",
    "_maybe_record",
]