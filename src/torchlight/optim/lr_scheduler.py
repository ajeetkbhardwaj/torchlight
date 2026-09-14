"""
Learning-rate schedulers.

A scheduler owns a *schedule*: given the epoch (or step count since the last
reset) it computes a multiplier delivered through the shared ``optimizer.lr``
attribute.  The design is deliberately minimal -- no per-parameter groups --
so it composes with every optimizer in :mod:`torchlight.optim` (``SGD``,
``Adam``, ``AdamW``).

Usage::

    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    for epoch in range(50):
        train(...)
        scheduler.step()          # decay the learning rate
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from .optimizer import Optimizer


class LRScheduler:
    """
    Base class for all schedules.

    Subclasses implement :meth:`get_lr` and the base keeps the epoch counter
    plus the initial learning rate(s) captured from ``optimizer``.

    Args:
        optimizer: any :class:`~torchlight.optim.Optimizer` with an ``lr``
            attribute (SGD / Adam / AdamW).
        last_epoch: the epoch the scheduler is created at (``-1`` = start).
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.base_lrs: List[float] = [float(optimizer.lr)]
        self.last_epoch = last_epoch

    def get_lr(self) -> List[float]:
        """Compute this epoch's learning rate(s) (implemented by subclasses)."""
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """Advance to the next epoch and apply the new learning rate."""
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        lrs = self.get_lr()
        self.optimizer.lr = lrs[0]

    def state_dict(self) -> Dict[str, object]:
        """Serialisable state for checkpointing the schedule."""
        return {
            "last_epoch": self.last_epoch,
            "base_lrs": list(self.base_lrs),
            "type": type(self).__name__,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore a :meth:`state_dict` snapshot."""
        self.last_epoch = int(state["last_epoch"])
        self.base_lrs = [float(lr) for lr in state["base_lrs"]]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(lr={self.optimizer.lr}, last_epoch={self.last_epoch})"


class StepLR(LRScheduler):
    """
    Decays the learning rate by ``gamma`` every ``step_size`` epochs:

    .. math:: \eta_t = \eta_0 \cdot \gamma^{\lfloor \text{epoch} / S \rfloor}
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        if step_size <= 0:
            raise ValueError("StepLR requires step_size > 0")
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        return [b * self.gamma ** (self.last_epoch // self.step_size) for b in self.base_lrs]


class MultiStepLR(LRScheduler):
    """
    Decays the learning rate by ``gamma`` at each epoch in ``milestones``.
    """

    def __init__(self, optimizer: Optimizer, milestones: Sequence[int], gamma: float = 0.1, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        hits = sum(1 for m in self.milestones if m <= self.last_epoch)
        return [b * self.gamma ** hits for b in self.base_lrs]


class ExponentialLR(LRScheduler):
    """
    Decays the learning rate by ``gamma`` every epoch:

    .. math:: \eta_t = \eta_0 \cdot \gamma^{\,\text{epoch}}
    """

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        return [b * self.gamma ** self.last_epoch for b in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """
    Cosine decay from ``eta_min`` up to the base learning rate over ``T_max``
    epochs:

    .. math::
        \eta_t = \eta_{\min} + \frac{\eta_0 - \eta_{\min}}{2}
                 \big(1 + \cos(\pi\, t / T_{\max})\big)
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        if T_max <= 0:
            raise ValueError("CosineAnnealingLR requires T_max > 0")
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return list(self.base_lrs)
        step = self.last_epoch
        frac = (1.0 + math.cos(math.pi * step / self.T_max)) / 2.0
        return [self.eta_min + (b - self.eta_min) * frac for b in self.base_lrs]


__all__ = ["LRScheduler", "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR"]