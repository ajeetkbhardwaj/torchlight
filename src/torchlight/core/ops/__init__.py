"""
Mathematical operator zoo.

Exports the scalar operators so the rest of the package (and user code) can do::

    from torchlight.core.ops import relu, map, sum

Everything here is *value-semantics* and framework-independent; they are the
building blocks that the numpy backend and the autograd functions build upon.
"""

from .scalar import *  # noqa: F401,F403
from .scalar import __all__  # noqa: F401