"""
``torchlight.scalar`` -- scalar-level autodiff.

Quick-start::

    from torchlight.scalar import Scalar, derivative_check

    x = Scalar(2.0); y = Scalar(3.0)
    z = x * y + x.sigmoid()
    z.backward()
    assert x.derivative is not None
    derivative_check(lambda x, y: x * y + x.sigmoid(), x, y)
"""

from .scalar import Scalar, ScalarHistory, ScalarLike, backpropagate, derivative_check
from .scalar_functions import ScalarFunction

__all__ = [
    "Scalar",
    "ScalarHistory",
    "ScalarLike",
    "ScalarFunction",
    "backpropagate",
    "derivative_check",
]