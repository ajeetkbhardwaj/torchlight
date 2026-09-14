"""
torchlight.tensor -- the public tensor API.

Exposes the :class:`Tensor` class and numpy-style construction helpers,
plus the basic concatenation / stacking / splitting operations:

    >>> import torchlight as tl
    >>> x = tl.tensor([[1., 2.], [3., 4.]])
    >>> tl.cat([x, x], dim=0).shape
    (4, 2)
"""

from .tensor import (
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

__all__ = [
    "Tensor",
    "tensor",
    "from_numpy",
    "zeros",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "arange",
    "cat",
    "stack",
    "split",
    "chunk",
]