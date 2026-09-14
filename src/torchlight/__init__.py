"""
Torchlight -- a lightweight, from-scratch deep learning framework.

Designed as a teaching-and-research PyTorch: same high-level ergonomics
(``tensor``, ``nn``, ``optim``, ``data``), a tiny autograd engine, and a
numpy-backed CPU backend with optional numba (CPU-JIT) and CUDA (numba-cuda)
accelerators.  Built for small-scale LLM experiments and for reading *every
line* of what makes deep learning tick.
"""

# Package metadata.
__version__ = "0.1.0"

# Core subpackages (imported eagerly so `import torchlight` is batteries-on).
from . import autograd, backends, core, data, jit, nn, optim, scalar, utils  # noqa: F401
from .core import *  # noqa: F401,F403
from .backends import CPUBackend, get_backend  # noqa: F401
from .scalar import Scalar, ScalarFunction, derivative_check  # noqa: F401
from .tensor import (  # noqa: F401
    Tensor,
    arange,
    cat,
    chunk,
    empty,
    from_numpy,
    full,
    ones,
    rand,
    randn,
    split,
    stack,
    tensor,
    zeros,
)

# Convenience aliases that match torch's top-level namespace.
from .tensor import tensor as _tensor  # noqa: F401


def __getattr__(name):
    """Lazily expose the numba / cuda backends (avoids importing numba until
    the user actually asks for an accelerator backend)."""
    if name in ("NumbaBackend", "CudaBackend"):
        from .backends import NumbaBackend, CudaBackend

        return NumbaBackend if name == "NumbaBackend" else CudaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "autograd",
    "backends",
    "core",
    "data",
    "jit",
    "nn",
    "optim",
    "scalar",
    "utils",
    "Tensor",
    "Scalar",
    "ScalarFunction",
    "derivative_check",
    "CPUBackend",
    "NumbaBackend",
    "CudaBackend",
    "get_backend",
]