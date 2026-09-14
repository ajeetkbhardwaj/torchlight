"""JIT tracing and serialization tests."""

import numpy as np
import pytest

import torchlight as tl
from torchlight.jit import trace
from torchlight.nn import Linear, ReLU
from torchlight.utils import (
    load_state,
    load_state_dict,
    save_model,
    save_state,
    state_dict,
)


class MLP(tl.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(2, 8)
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        return self.fc2(ReLU()(self.fc1(x)))


def test_trace_captures_expected_ops():
    model = MLP()
    xin = tl.tensor(np.random.rand(4, 2).astype(np.float32))
    tape, _ = trace(model, xin)
    counts = tape.count_ops()
    assert counts["MatMul"] == 2
    assert counts["Add"] == 2
    assert counts["ReLU"] == 1


def test_tape_replay_matches_eager_forward():
    model = MLP()
    xin = tl.tensor(np.random.rand(4, 2).astype(np.float32))
    tape, _ = trace(model, xin)
    new = tl.tensor(np.random.rand(4, 2).astype(np.float32))
    replayed = tape.run(new)[0]
    direct = model(new)
    assert np.allclose(replayed.to_numpy(), direct.to_numpy())


def test_tape_replay_needs_same_input_count():
    model = MLP()
    tape, _ = trace(model, tl.tensor(np.random.rand(2, 2).astype(np.float32)))
    with pytest.raises(ValueError):
        tape.run()


def test_state_dict_names_and_roundtrip(tmp_path):
    m1 = MLP()
    sd = state_dict(m1)
    assert set(sd) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    m2 = MLP()
    load_state_dict(m2, sd)
    assert np.allclose(m1.fc1.weight.to_numpy(), m2.fc1.weight.to_numpy())
    assert m2.fc1.weight.value.is_leaf()

    p = tmp_path / "m.npz"
    save_state(m2, str(p))
    m3 = MLP()
    load_state(m3, str(p))
    assert np.allclose(m3.fc1.weight.to_numpy(), m1.fc1.weight.to_numpy())


def test_strict_state_dict_detects_mismatch(tmp_path):
    m = MLP()
    with pytest.raises(ValueError):
        load_state_dict(m, {"nope": np.zeros(4)})


def test_save_load_model(tmp_path):
    m = MLP()
    p = tmp_path / "m.pkl"
    save_model(m, str(p))
    from torchlight.utils import load_model

    m2 = load_model(str(p))
    assert np.allclose(m2.fc1.weight.to_numpy(), m.fc1.weight.to_numpy())