# Tutorial 1: Tensors

Torchlight's `Tensor` is the central object: an N-dimensional array of
`float32` with automatic shape/strides bookkeeping and reverse-mode auto-diff
when you ask for it.

## Creating tensors

Create tensors from Python lists or numpy arrays:

```python
import numpy as np
import torchlight as tl

a = tl.tensor([1.0, 2.0, 3.0])                 # from a list -> shape (3,)
b = tl.tensor([[1.0, 2.0], [3.0, 4.0]])        # shape (2, 2)
c = tl.from_numpy(np.arange(6).reshape(2, 3))  # from numpy -> shape (2, 3)
```

Factory functions cover the common allocations:

```python
tl.zeros((2, 3))          # all zeros
tl.ones((2, 3))           # all ones
tl.full((2, 3), 0.5)      # all 0.5
tl.empty((2, 3))          # uninitialised
tl.randn((2, 3))          # standard normal
tl.rand((2, 3))           # uniform [0, 1)
tl.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
```

Every factory accepts `requires_grad=True` and a `device=` backend hint
(see [Tutorial 5 — Backends](./05-backends.md)):

```python
x = tl.zeros((3, 4), requires_grad=True, device="cpu")
```

## Shape, strides, and views

A tensor knows its shape and total number of elements:

```python
x = tl.tensor(np.arange(12).reshape(3, 4))
x.shape               # (3, 4)
x.size                # 12
x.dims                # 2
```

Views share the underlying storage — no copy:

```python
y = x.view(4, 3)            # reshape view (contiguous)
t = x.transpose(0, 1)       # (4, 3) strided view, no copy
p = x.permute(1, 0)         # same as transpose
r = x.reshape(2, 6)         # materialises only when storage is non-contiguous
```

`contiguous()` forces a dense copy when the view is strided:

```python
t = x.transpose(0, 1)
t.is_contiguous               # False for a transposed view
dense = t.contiguous()        # dense copy
```

## Broadcasting

Binary ops broadcast following numpy's rules (trailing dimensions align,
size-1 dims expand):

```python
a = tl.tensor([[1.0], [2.0]])     # (2, 1)
b = tl.tensor([[10.0, 20.0]])     # (1, 2)
s = a + b                         # (2, 2): [[11, 21], [12, 22]]
```

[`TensorData`](../api/core.md) implements the index arithmetic; strided
broadcast views use stride-zero, so no data is ever copied.

## Elementwise ops, reductions, and more

```python
x = tl.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
x.abs().to_numpy()        # [2, 1, 0, 1, 2]
x.sigmoid()               # logistic, numerically stable
x.relu()                  # max(x, 0)
x.sqrt()                  # guarded for x <= 0 (returns 0)
x.exp(); x.log(); x.tanh()

m = tl.tensor([[1.0, 2.0], [3.0, 4.0]])
m.sum()                   # shape (1,) = 10
m.sum(dim=0)              # shape (1, 2)
m.mean()                  # 2.5
m.max(dim=1)              # shape (2, 1)
m.var(); m.std()          # population variance / std
m.clamp(2.0, 3.0)         # elementwise clip

c1 = tl.tensor([2.0, 3.0])
c2 = tl.tensor([1.0, 5.0])
c1.is_close(c2)           # |a-b| < 1e-2, returns 0/1 tensor
```

Matrix multiplication uses the `@` operator (with batched support):

```python
A = tl.tensor(np.random.randn(3, 4))
B = tl.tensor(np.random.randn(4, 5))
C = A @ B                 # (3, 5)
```

## Getting numbers back out

```python
t = tl.tensor([[1.0, 2.0], [3.0, 4.0]])
t.to_numpy()              # dense numpy copy
t[0, 1]                   # scalar float (full-index read)
t.item(0, 1)              # same, explicit
```

## In-place mutation helpers

These write into the existing buffer:

```python
t = tl.zeros((3,))
t.fill_(2.5)              # all entries 2.5
t.uniform_(-1.0, 1.0)     # in-place U(-1, 1)
t.normal_(0.0, 1.0)       # in-place N(0, 1)
t.zeros_(); t.ones_()     # in-place reset
```

## The storage layer

Underneath every `Tensor` is a `TensorData` in `torchlight.core`: a flat one-
dimensional `float32` buffer plus shape/strides. The backend kernels (numpy
ufuncs, numba/cuda kernels) operate directly on that layout via
`numpy_view()` — this is what keeps the whole engine free of Python loops.
See [TensorData API](../api/core.md#tensordata) for the low-level view helpers
(`permute`, `broadcast_to`, `numpy_view`, `to_numpy`).

## Next

Continue to [Tutorial 2 — Autograd](./02-autograd.md) to see how gradients
flow through the graph.
