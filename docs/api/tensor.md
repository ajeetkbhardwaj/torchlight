# API Reference: tensor

`torchlight.tensor` defines `Tensor` and the factory functions.

```python
import torchlight as tl
```

## Tensor

```python
class Tensor(data: TensorData, history: Optional[History] = None, backend=None)
```

The core array type. Backed by a numpy `TensorData`; ops route through the
backend (`CPU`, `numba`, or `cuda`).

### Attributes

| Attribute | Meaning |
|-----------|---------|
| `shape` | `tuple[int, ...]` |
| `size` | number of elements |
| `dims` | number of dimensions |
| `backend` | backend instance in use |
| `device` | `Device` of the backend |
| `requires_grad` | whether leaf accumulates gradients |
| `grad` | accumulated gradient tensor (or `None`) |

### Elementwise / reductions

| Method | Description |
|--------|-------------|
| `abs()`, `sqrt()`, `exp()`, `log()`, `sigmoid()`, `relu()`, `tanh()` | elementwise |
| `sum(dim=None)` | reduce (dim kept as size 1; all if `None`) |
| `mean(dim=None)` | arithmetic mean |
| `max(dim=None)`, `min(dim=None)` | reduction |
| `var(dim=None)`, `std(dim=None)` | population variance / std |
| `clamp(lo, hi)` | elementwise clip |
| `is_close(other)` | `|a-b| < 1e-2` → 0/1 tensor |
| `all()`, `any()` | logical reductions |

### Shape / storage

| Method | Description |
|--------|-------------|
| `view(*shape)` | reshape view (no copy when contiguous) |
| `reshape(*shape)` | view or materialised copy |
| `unsqueeze(dim)` | insert a size-1 axis (negative `dim` from the end) |
| `transpose(dim0, dim1)` / `permute(*order)` | strided view reorder |
| `flatten()` | collapse to 1-D |
| `contiguous()` | dense copy if strided |
| `to_numpy()` | dense numpy array |
| `item(*idx)` | scalar float at index |
| `clone()` | fresh copy in the graph |
| `detach()` | new tensor, same data, no grad |

### Indexing / gathering

| Method | Description |
|--------|-------------|
| `x[i]`, `x[1:3]`, `x[:, None]`, `x[...]` | advanced indexing (differentiable) |
| `index_select(dim, index)` | select rows/columns along `dim` |
| `cat(tensors, dim=0)` | concatenate along an axis |
| `stack(tensors, dim=0)` | concatenate along a new axis |
| `split(tensor, sizes, dim=0)` | split into chunks |
| `chunk(tensor, chunks, dim=0)` | split into equal-ish chunks |

`__getitem__` supports full-Python-style keys: ints, slices (with steps and
negative steps), `None` for new axes, and `...` (ellipsis), and is
fully differentiable:

```python
x = tl.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

x[0]                 # tensor([1., 2., 3.])  — partial int → one row
x[:, 1]              # tensor([2., 5.])
x[1, 2]              # 6.0                    — full int tuple → scalar float
x[::-1, 1:]          # reversed rows, cols 1..  (supports negative steps)
x[:, None]           # shape (2, 1, 3)          — new axis
x[..., -1]           # ellipsis → last column    tensor([3., 6.])
```

`unsqueeze`, `index_select`, `cat`, `stack`, `split`, and `chunk` are also
available at module level:

```python
tl.cat([a, b], dim=0)
tl.stack([a, b, c], dim=-1)
tl.split(x, 1, dim=1)   # chunks of size 1 along dim 1
tl.chunk(x, 2, dim=0)   # 2 equal-ish chunks along dim 0
```

### In-place

| Method | Description |
|--------|-------------|
| `fill_(v)` | set all entries to `v` |
| `zeros_()`, `ones_()` | reset in place |
| `uniform_(lo, hi)`, `normal_(mean, std)` | in-place random |
| `requires_grad_(bool)` | toggle grad tracking |
| `zero_grad_()` | zero the accumulated gradient |
| `rand_like()` | new random tensor, same shape |

### Autograd

| Method | Description |
|--------|-------------|
| `backward(deriv=1.0)` | run reverse-mode autodiff |

### Operators

Supported arithmetic operators: `+ - * / ** @` (with NumPy broadcasting),
plus Python scalar comparisons through the backend (`<`, `>`, `==`).

## Factories

```python
tensor(data, backend=None, requires_grad=False, device=None)
from_numpy(arr, backend=None, requires_grad=False, device=None)
zeros(shape, backend=None, requires_grad=False, device=None)
ones(shape, backend=None, requires_grad=False, device=None)
empty(shape, backend=None, requires_grad=False, device=None)
full(shape, value, backend=None, requires_grad=False, device=None)
rand(shape, backend=None, requires_grad=False, device=None)     # U(0,1)
randn(shape, backend=None, requires_grad=False, device=None)    # N(0,1)
arange(start, stop=None, step=1.0, requires_grad=False, device=None)
```

`device` accepts `"cpu"`, `"cuda"`, or a `Device`; it selects the
backend via `get_backend(device)`.

```python
tl.tensor([[1.0, 2.0], [3.0, 4.0]])
tl.zeros((3, 4), requires_grad=True)
tl.arange(0, 5, 2)               # [0, 2, 4]
tl.randn((2, 2), device="cuda")
```