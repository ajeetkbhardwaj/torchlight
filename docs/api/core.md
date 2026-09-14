# API Reference: core

`torchlight.core` is the raw foundation: the storage container
(`TensorData`), devices, broadcasting/shape helpers, and the scalar op
spec (`ops`).

```python
from torchlight.core import TensorData, Device, cpu, cuda, resolve_device
from torchlight.core.tensor_data import shape_broadcast, strides_from_shape
from torchlight.core.ops import scalar
```

## Device

```python
class Device(type: str, index: int = 0)
```

A lightweight descriptor of where a tensor computes.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `type` | `str` | `"cpu"` or `"cuda"` |
| `index` | `int` | device index (unused today) |
| `is_cpu` | `bool` | `type == "cpu"` |
| `is_cuda` | `bool` | `type == "cuda"` |

### Predefined devices

| Value | Meaning |
|-------|---------|
| `cpu` | CPU (numpy backend) |
| `cuda` | NVIDIA GPU (numba-cuda backend) |

### `resolve_device(device)`

```python
def resolve_device(device: Optional[Device | str | None]) -> Device
```

Normalize `None` / string / `Device` into a `Device`. Default device follows
the `TORCHLIGHT_DEVICE` environment variable (default `"cpu"`). Unknown
devices raise `ValueError`.

```python
resolve_device(None)              # cpu (or TORCHLIGHT_DEVICE value)
resolve_device("cuda")            # device('cuda:0')
resolve_device(cpu)               # device('cpu:0')
```

## TensorData

```python
class TensorData(storage, shape, strides=None, device=None)
```

The raw data container: a flat one-dimensional numpy `float32` buffer plus
shape/strides. Computes go through the backends, which operate directly on
this layout via `numpy_view()`.

### Construction

```python
TensorData(np.arange(6, dtype=np.float32), (2, 3))
```

`device` defaults to CPU; non-CPU devices raise `ValueError` (storage is
always host numpy — backends transfer per-op).

### Properties

| Attribute | Type | Meaning |
|-----------|------|---------|
| `shape` | `tuple[int, ...]` | logical shape |
| `strides` | `tuple[int, ...]` | strides (in elements) |
| `storage` | `np.ndarray` | flat float32 buffer |
| `dims` | `int` | number of dimensions |
| `size` | `int` | total number of elements |
| `dtype` | `np.dtype` | numpy dtype of the buffer |

### View-producing operations

| Method | Description |
|--------|-------------|
| `permute(*order)` | reorder dimensions (share storage, no copy) |
| `broadcast_to(shape)` | strided broadcast view (stride-0 dims) |
| `reshape(shape)` | reuse storage if contiguous, else materialise |
| `contiguous()` | dense copy if not already contiguous |
| `numpy_view()` | strided numpy view (no copy) on the buffer |
| `to_numpy()` | dense contiguous numpy copy |

### Indexing

| Method | Description |
|--------|-------------|
| `get(index)` | read an element (full index tuple) |
| `set(index, val)` | write an element (CPU only) |
| `indices()` | iterator over all index tuples |

### Allocators (static)

```python
TensorData.zeros(shape)  → TensorData
TensorData.ones(shape)   → TensorData
TensorData.empty(shape)  → TensorData
```

## Shape & stride helpers

```python
def shape_broadcast(shape1, shape2) -> UserShape
```

NumPy broadcasting rules for two shapes; raises on incompatibility.

```python
def strides_from_shape(shape) -> UserStrides
```

Row-major strides for a contiguous shape.

## Index arithmetic

```python
index_to_position(index, strides) -> int
to_index(shape, index) -> Tuple[int, ...]
broadcast_index(shape, shape1, strides1, strides2) -> Tuple[int, ...]
```

Pure-Python index <-> flat-position conversions used by the kernels.

## Scalar ops (`torchlight.core.ops`)

The mathematical op *spec*, one function per operation. Autograd functions,
CPU arrays, and numba/cuda kernels all derive from these definitions.

```python
from torchlight.core.ops import scalar

scalar.sigmoid(x)     # stable logistic
scalar.relu(x)
scalar.log(x); scalar.exp(x)
scalar.inv(x)
scalar.add(a, b); scalar.mul(a, b)
scalar.relu_back(x, d); scalar.log_back(x, d)
scalar.is_close(a, b, tol=...)
```