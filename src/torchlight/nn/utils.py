"""
``torchlight.nn.utils`` -- training utilities (mirrors ``torch.nn.utils``).

Only the helpers that generalise across optimizers / models live here --
currently gradient clipping.  Everything is implemented with the primitive
tensor ops, so it runs unchanged on the CPU, numba and CUDA backends.
"""

from __future__ import annotations

import math
from typing import Iterable, Union

from .module import Parameter

ParameterSet = Union[Iterable[Parameter], Parameter]


def _trip(parameters: ParameterSet):
    """Normalise an iterable-or-single into the *parameters that have a grad*."""
    if isinstance(parameters, Parameter):
        parameters = (parameters,)
    for p in parameters:
        value = getattr(p, "value", None)
        grad = getattr(value, "grad", None) if value is not None else None
        if grad is None:
            continue
        yield p, grad.detach()


def clip_grad_norm_(parameters: ParameterSet, max_norm: float, norm_type: float = 2.0) -> float:
    r"""
    Clip the gradient norms of ``parameters`` in place (torch-compatible).

    The total norm is :math:`\left(\sum_p \|g_p\|_p^{\,n}\right)^{1/n}` (for
    ``norm_type == inf`` it is the max absolute gradient over all
    parameters).  If that exceeds ``max_norm`` every gradient is rescaled by
    ``max_norm / (total_norm + 1e-6)``; otherwise nothing changes.

    Returns the pre-clip total norm.  The result is identical on every
    backend because it is built from the common ``abs`` / ``pow`` / ``sum``
    primitives.
    """
    grads: list = []
    total: float = 0.0
    for p, g in _trip(parameters):
        grads.append((p, g))
        if math.isinf(norm_type):
            total = max(total, float(g.abs().max().item()))
        else:
            total += float((g.abs() ** norm_type).sum().item())

    if not grads:
        return 0.0

    if math.isinf(norm_type):
        total_norm = total
    else:
        total_norm = total ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p, g in grads:
            p.value.grad = g * clip_coef
    return float(total_norm)