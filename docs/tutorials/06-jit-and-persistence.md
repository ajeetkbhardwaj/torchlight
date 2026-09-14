# Tutorial 6: JIT tracing and Persistence

Torchlight ships two small conveniences on top of autograd: an **op tracer**
that records a forward pass as a tape, and **serialization** helpers to save
and load models.

## JIT tracing

`trace()` runs a callable on sample inputs while recording every
`Function` op that fires — producing a synthetic `GraphTape`:

```python
from torchlight.jit import trace, count_ops, exec_tape

def forward(x, w):
    return (x @ w).relu().sum()

x = tl.tensor([[1.0, 2.0]])
w = tl.tensor([[2.0], [3.0]])

tape, outputs = trace(forward, x, w)
len(tape.records)              # number of recorded ops
```

## Counting ops

`count_ops()` tallies how many times each function appears (including
internal bookkeeping ops like `Contiguous`/`View`):

```python
ops = count_ops(tape)
ops              # e.g. {'MatMul': 1, 'ReLU': 1, 'Sum': 1, 'Contiguous': 1, 'View': 1}
```

Filter to the ops you care about to compare architectures:

```python
user_ops = {k: v for k, v in ops.items() if k not in {"Contiguous", "View"}}
```

## Replaying the tape without autograd

`exec_tape()` re-runs the recorded ops on new input tensors **without
rebuilding the autograd graph** — a pure forward re-execution:

```python
x2 = tl.tensor([[5.0, 6.0]])
result = exec_tape(tape, x2, w)[0]     # first output tensor, same shape as original
```

Because the tape stores the *function classes* and their inputs, you can
even inspect it:

```python
for record in tape.records:
    print(record.fn.__name__)     # the Function class used
```

## Saving and loading models

### Whole-model save/load

Save a pickled `Module` (gracefully handles custom classes), then reload:

```python
from torchlight.utils import save_model, load_model

model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
save_model(model, "model.pkl")
loaded = load_model("model.pkl")     # a fresh, pickled copy
```

### State dicts (portable weight save)

Save just the named parameters/state as a `.npz` (numpy) file — no pickling:

```python
from torchlight.utils import save_state, load_state, state_dict, load_state_dict

# Save/load a model's state to a file
save_state(model, "state.npz")
new_model = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
load_state(new_model, "state.npz")   # exact same weights now

# Or manipulate dicts directly
state = state_dict(model)            # {'linear0.weight': np.ndarray, ...}
load_state_dict(model, state, strict=True)
```

`strict=True` (default) raises if a name in the dict is missing from the
model, or vice versa.

## Example: full save/load round-trip

```python
import numpy as np
import torchlight as tl
from torchlight.nn import Linear, Sequential, ReLU
from torchlight.utils import save_state, load_state

np.random.seed(0)
m1 = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
save_state(m1, "/tmp/example_state.npz")

np.random.seed(1)
m2 = Sequential(Linear(2, 8), ReLU(), Linear(8, 1))
load_state(m2, "/tmp/example_state.npz")

x = tl.randn((5, 2))
assert (m1(x) - m2(x)).abs().sum().to_numpy().ravel()[0] < 1e-4
```

!!! note
    `save_model`/`load_model` use Python pickling — fine for local models.
    `save_state`/`load_state` store numpy arrays (`.npz`), which is the
    recommended portable format.

## See also

- [JIT API reference](../api/jit.md)
- [Utils API reference](../api/utils.md)