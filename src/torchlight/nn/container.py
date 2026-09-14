"""
``Sequential``: pipe modules together.

Modules are stored (and printed) in the order they were passed; calling a
``Sequential`` runs them in that order.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .module import Module


class Sequential(Module):
    """A container that forwards inputs through a list of modules in order."""

    def __init__(self, *modules: Module):
        super().__init__()
        # Rebuild as an ordered registry so Module traversal sees them.
        for i, mod in enumerate(modules):
            self._modules[str(i)] = mod

    @property
    def modules_list(self) -> List[Module]:
        """The ordered list of child modules."""
        return [self._modules[str(i)] for i in range(len(self._modules))]

    def forward(self, input: Any) -> Any:
        for mod in self.modules_list:
            input = mod(input)
        return input

    def __repr__(self) -> str:
        lines = []
        for i, mod in self.modules_list:
            lines.append(f"  ({i}): {repr(mod)}")
        body = "\n".join(lines)
        return f"Sequential(\n{body}\n)" if body else "Sequential()"