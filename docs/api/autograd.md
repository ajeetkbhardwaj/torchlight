# API Reference: autograd

`torchlight.autograd` is the reverse-mode auto-diff engine.

```python
from torchlight.autograd import Function, Context, History
from torchlight.autograd import backpropagate, topological_sort, central_difference
```

## Function

```python
class Function(...)
```

The abstract op pair. Subclasses implement `forward(ctx, *inputs)` and
`backward(ctx, grad_output)`; the framework handles graph wiring. Instances
are never created by users — call them like `Add.apply(a, b)`.

```python
from torchlight.autograd import Function

class Square(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a.backend.zip(scalar.mul)(a, a)

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return a.backend.zip(scalar.mul)(a, g) * 2
```

## Context

```python
class Context(no_grad=False, saved_values=())
```

Scratchpad passed between a function's `forward` and `backward`.

| Method | Description |
|--------|-------------|
| `save_for_backward(*values)` | stash tensors for the backward pass |
| `saved_tensors` | the stashed values (`ctx.save_for_backward` + `ctx.saved_tensors`) |

## History

```python
class History(last_fn=None, ctx=None, inputs=())
```

One edge of the compute graph.

| Attribute | Meaning |
|-----------|---------|
| `last_fn` | the `Function` class that produced the tensor (or `None` for a leaf) |
| `ctx` | the forward's `Context` (saved values) |
| `inputs` | parent tensors |

## backpropagate

```python
def backpropagate(variable: Variable, deriv: Any) -> None
```

Run reverse-mode differentiation from `variable` back to the leaves,
accumulating `.grad` into every non-constant leaf.

## topological_sort

```python
def topological_sort(variable: Variable) -> List[Variable]
```

Return the non-constant graph nodes from leaves to root.

## central_difference

```python
def central_difference(f, *vals, arg=0, epsilon=1e-6) -> float
```

Central-difference numerical derivative of `f` w.r.t. `vals[arg]` — used by
the test suite to gradcheck every op.

## The built-in ops

Every op used by `Tensor` is defined in `torchlight.autograd.functions`
(e.g. `Add`, `Mul`, `Sub`, `Div`, `Pow`, `MatMul`, `Sum`, `Map`,
`Log`, `Exp`, `Sigmoid`, `ReLU`, `ReluBack`, `LogBack`, `Inv`, `InvBack`,
`Sqrt`, `Abs`, `Tanh`, `Permute`, `BroadcastView`, `Gather`, `Max`). They
are called through `Tensor` methods and through
`torchlight.nn.functional`.