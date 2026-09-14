"""
torchlight.optim -- gradient-based optimizers and LR schedules.

* :class:`SGD`   -- with optional momentum / weight decay.
* :class:`Adam`, :class:`AdamW`.
* :mod:`lr_scheduler` -- StepLR / MultiStepLR / ExponentialLR / CosineAnnealingLR.
"""

from .adam import Adam, AdamW
from .lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LRScheduler,
    MultiStepLR,
    StepLR,
)
from .optimizer import Optimizer
from .sgd import SGD

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "LRScheduler",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
]