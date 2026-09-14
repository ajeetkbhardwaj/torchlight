"""
torchlight.autograd -- automatic differentiation.

* :mod:`autodiff`   -- Context, History, topological sort, backprop.
* :mod:`functions`  -- every differentiable operation as a :class:`Function`.
"""

from . import autodiff, functions  # noqa: F401
from .autodiff import Context, History, backpropagate, central_difference, topological_sort
from .functions import *  # noqa: F401,F403
from .functions import Function

__all__ = [
    "Function",
    "Context",
    "History",
    "backpropagate",
    "topological_sort",
    "central_difference",
    "autodiff",
    "functions",
]