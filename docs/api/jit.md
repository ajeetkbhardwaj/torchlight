# API Reference: jit

`torchlight.jit` records and replays forward passes as op "tapes".

```python
from torchlight.jit import trace, count_ops, exec_tape
```

## trace

```python
def trace(fn: Callable[..., Any], *sample_inputs: Tensor) -> Tuple[GraphTape, Tuple[Tensor, ...]]
```

Run `fn(*sample_inputs)` while recording every autograd op that fires.
Returns the `GraphTape` and the recorded output tensors.

```python
def forward(x, w):
    return (x @ w).relu().sum()

tape, outputs = trace(forward, x, w)
```

## GraphTape

```python
class GraphTape(records: List[OpRecord], sample_inputs, outputs)
```

| Attribute | Meaning |
|-----------|---------|
| `records` | `OpRecord` list — one per op |
| `sample_inputs` | the input tensors the tape was captured with |
| `outputs` | recorded output tensors |

### OpRecord

```python
class OpRecord(fn, inputs, output)
```

A single recorded op.

| Attribute | Meaning |
|-----------|---------|
| `fn` | the `Function` class |
| `inputs` | parent tensors |
| `output` | output tensor |

## count_ops

```python
def count_ops(tape: GraphTape) -> Dict[str, int]
```

Count how many times each function runs during the forward pass (includes
internal `Contiguous`/`View` records).

```python
ops = count_ops(tape)   # {'MatMul': 1, 'ReLU': 1, 'Sum': 1, 'Contiguous': 1, 'View': 1}
```

## exec_tape

```python
def exec_tape(tape: GraphTape, *run_inputs: Tensor) -> Tuple[Tensor, ...]
```

Re-run the recorded ops on new inputs **without rebuilding autograd**.

```python
results = exec_tape(tape, new_x, new_w)   # same shape as the original outputs
```

## Tracer

```python
class Tracer()
```

The low-level recorder (used by `trace`). Registers an op callback via
`torchlight.autograd.set_op_callback` and restores it afterwards.