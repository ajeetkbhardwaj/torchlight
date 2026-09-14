"""
Tests for the learning-rate schedulers (StepLR, MultiStepLR, ExponentialLR,
CosineAnnealingLR): schedule values, epoch bookkeeping, and serialisation.
"""
from __future__ import annotations

import math

import pytest

import torchlight as tl
from torchlight.nn import Linear
from torchlight.optim import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    SGD,
    StepLR,
)


@pytest.fixture
def opt():
    model = Linear(2, 2)
    return SGD(model.parameters(), lr=0.5)


class TestStepLR:
    def test_schedule(self, opt):
        sched = StepLR(opt, step_size=3, gamma=0.5)
        lrs = []
        for _ in range(10):
            sched.step()
            lrs.append(opt.lr)
        expected = [0.5] * 3 + [0.25] * 3 + [0.125] * 3 + [0.0625]
        assert lrs == expected

    def test_invalid(self, opt):
        with pytest.raises(ValueError):
            StepLR(opt, step_size=0)


class TestExponentialLR:
    def test_schedule(self, opt):
        sched = ExponentialLR(opt, gamma=0.9)
        for i in range(5):
            sched.step()      # epochs 0, 1, 2, 3, 4
            assert opt.lr == pytest.approx(0.5 * 0.9 ** i)


class TestMultiStepLR:
    def test_schedule(self, opt):
        sched = MultiStepLR(opt, milestones=[2, 4])
        lrs = []
        for _ in range(6):
            sched.step()
            lrs.append(round(opt.lr, 4))
        # gammas apply at epochs 2 and 4
        assert lrs == [0.5] * 2 + [0.05] * 2 + [0.005] * 2


class TestCosineAnnealingLR:
    def test_extremes(self, opt):
        sched = CosineAnnealingLR(opt, T_max=4)
        sched.step()          # epoch 0 -> base
        assert opt.lr == pytest.approx(0.5)
        for _ in range(4):
            sched.step()      # epoch 1..4 -> decays towards eta_min
        # at epoch 4 the cosine has completed a half-cycle: lr ~= eta_min
        assert opt.lr == pytest.approx(0.0, abs=1e-6)

    def test_middle_value(self, opt):
        sched = CosineAnnealingLR(opt, T_max=4)
        for _ in range(2):
            sched.step()      # epoch 1
        expect = 0.5 * (1.0 + math.cos(math.pi * 1 / 4)) / 2.0
        assert opt.lr == pytest.approx(expect)

    def test_eta_min(self, opt):
        sched = CosineAnnealingLR(opt, T_max=2, eta_min=0.1)
        for _ in range(3):
            sched.step()      # epoch 2 == T_max -> minimum reached
        assert opt.lr == pytest.approx(0.1)


class TestStateDict:
    def test_roundtrip(self, opt):
        sched = StepLR(opt, step_size=2, gamma=0.5)
        for _ in range(3):
            sched.step()
        alt_opt = SGD(Linear(2, 2).parameters(), lr=0.5)
        alt = StepLR(alt_opt, step_size=2, gamma=0.5)
        alt.load_state_dict(sched.state_dict())
        assert alt.last_epoch == sched.last_epoch
        alt.step()
        sched.step()
        assert alt_opt.lr == opt.lr