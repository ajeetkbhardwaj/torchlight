"""
Optimizer base class.

Optimizers dance with :class:`Module` parameters in three steps:

1. ``optimizer.zero_grad()`` -- clear each parameter's ``grad``.
2. forward + ``loss.backward()`` -- autograd fills in the gradients.
3. ``optimizer.step()`` -- update each parameter from its gradient.

Subclasses implement the actual update rule in :meth:`step`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..nn.module import Parameter


class Optimizer:
    """
    Base optimizer.

    Args:
        parameters: an iterable of :class:`Parameter` to optimise (in
            practice ``model.parameters()``).
        defaults: dictionary of default hyperparameter values.
    """

    def __init__(self, parameters: Iterable[Parameter], defaults: Optional[Dict[str, Any]] = None):
        self.parameters: List[Parameter] = list(parameters)
        if not self.parameters:
            raise ValueError("Optimizer got an empty parameter list")
        self.defaults: Dict[str, Any] = {} if defaults is None else dict(defaults)
        #: Per-parameter mutable state (momentum buffers, Adam moments, ...).
        self.state: Dict[int, Dict[str, Any]] = {}

    def zero_grad(self) -> None:
        """Set ``grad`` to ``None`` on every parameter (accumulation reset)."""
        for p in self.parameters:
            if p.value is not None and hasattr(p.value, "grad"):
                p.value.grad = None

    def step(self) -> None:
        """Perform a single optimisation step (implemented by subclasses)."""
        raise NotImplementedError

    # -- Small helpers --------------------------------------------------------
    def _grad(self, p: Parameter):
        """Return the parameter's gradient tensor (or ``None``)."""
        if p.value is None:
            return None
        return getattr(p.value, "grad", None)  # raw accumulated gradient

    def _param_state(self, p: Parameter) -> Dict[str, Any]:
        """Lazily create and return the per-parameter state dict."""
        if id(p) not in self.state:
            self.state[id(p)] = {}
        return self.state[id(p)]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        defaults = ", ".join(f"{k}={v}" for k, v in self.defaults.items())
        return f"{type(self).__name__}({defaults})"