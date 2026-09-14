# Tutorial 2: Autograd

Torchlight uses **reverse-mode automatic differentiation** over a dynamic
computation graph — the same system that powers training in every deep learning
framework, built from scratch in ~200 lines.

## How gradients work

Ask for gradients by passing `requires_grad=True` to any factory:

```python
import torchlight as tl

x = tl.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()   # y = x1^2 + x2^2
y.backward()          # seed gradient = 1.0
x.grad                # [4.0, 6.0]  (dy/dx = 2x)
```

## The computation graph

Every intermediate tensor carries a `History` node recording *which*
function produced it, and from *what inputs*:

```python
a = tl.tensor([1.0, 2.0], requires_grad=True)
b = a + a          # History: Add, parents=(a, a)
c = b * a          # History: Mul, parents=(b, a)
c.backward()
a.grad             # [4.0, 8.0]  —  (d(b*a)/da = b + a*1 = 2a + a)
```

Nodes are cleaned up automatically when nothing references them, so no
manual graph management is needed.

## `backward()`

```python
loss = model(x).mse_loss(target)
loss.backward()          # populates .grad on every leaf with requires_grad=True
```

The root tensor receives `deriv = 1.0` (or any explicit gradient you pass
as the first argument).

## Leaf vs. intermediate tensors

Only **leaf** tensors (created by the user, not by a function) accumulate
gradients:

```python
a = tl.tensor([1.0, 2.0], requires_grad=True)  # leaf
b = a * 2                                       # intermediate
b.backward()
a.grad    # [2.0, 4.0]
b.grad    # AttributeError — not a leaf
```

Detach a leaf from the graph:

```python
c = a.detach()    # shares storage, no grad
```

## Detach, zero_grad, cloning

```python
x.zero_grad_()                          # in-place: sets grad to zeros
detached = x.detach()                   # new tensor, same data, no graph
cloned = x.clone()                      # fresh copy in the graph
```

## Checking gradients manually

Use `central_difference()` from `torchlight.autograd` to validate any
gradient analytically computed by `backward()`:

```python
from torchlight.autograd import central_difference

def f(x): return (x ** 2).sum()
x_val = [3.0, 4.0]
for i in range(len(x_val)):
    print(central_difference(lambda *xs: f(tl.tensor(xs)), *x_val, arg=i))
# 5.999998...  7.999998...   (true grad = [6.0, 8.0])
```

## Autograd internals (under the hood)

| Class | Role |
|-------|------|
| `Context` | Scratchpad a `Function` uses to stash values for `backward()`. |
| `History` | Records which function built a tensor, with its context and parent tensors. |
| `Function` | The `forward`/`backward` pair (e.g. `Add`, `Mul`, `MatMul`). |
| `backpropagate(root, deriv)` | Walks the graph in reverse, accumulating `.grad` into leaves. |

The engine is a standard stack: `topological_sort` orders the graph from
leaves to root, then `backpropagate` walks it backwards calling each
function's `chain_rule`.

See [Autograd API](../api/autograd.md) for full signatures.

## Next

Learn to compose gradient-tracked ops into [neural networks](./03-neural-networks.md).