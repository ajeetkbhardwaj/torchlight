# API Reference: data

`torchlight.data` provides datasets, batches, and synthetic problems.

```python
from torchlight.data import Dataset, TensorDataset, DataLoader, make_synthetic
```

## Dataset

```python
class Dataset()
```

Abstract base class: implement `__len__` and `__getitem__(idx)`.

## TensorDataset

```python
class TensorDataset(*tensors)
```

A fixed-size dataset of parallel array-likes (tensors, numpy arrays, or
lists), converted lazily to Tensors per element.

| Method | Description |
|--------|-------------|
| `len(ds)` | number of elements |
| `ds[i]` | `tuple[Tensor, ...]` — one tensor per field |
| `to_batch()` | whole dataset as `tuple[Tensor, ...]` shaped `(N, ...)` |

```python
ds = TensorDataset(tl.tensor([1., 2.]), tl.tensor([0, 1]))
ds[0]        # (tensor([1.]), tensor([0.]))
ds.to_batch()  # (tensor([1., 2.]), tensor([0., 1.]))
```

## DataLoader

```python
class DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, seed=None)
```

Iterate over a dataset in batches.

```python
for batch_x, batch_y in DataLoader(ds, batch_size=16, shuffle=True):
    ...
```

| Argument | Meaning |
|----------|---------|
| `dataset` | any `Dataset` (e.g. `TensorDataset`) |
| `batch_size` | elements per batch |
| `shuffle` | randomize order each epoch |
| `drop_last` | drop the last incomplete batch |
| `seed` | RNG seed for shuffling |

## Samplers

```python
from torchlight.data import SequentialSampler, RandomSampler, BatchSampler
```

| Class | Description |
|-------|-------------|
| `SequentialSampler(n)` | indices `0..n-1` in order |
| `RandomSampler(n, seed=None)` | indices in random order |
| `BatchSampler(sampler, batch_size, drop_last=False)` | group indices into batches |

## Synthetic problems

### make_synthetic

```python
def make_synthetic(name: str, n=100, seed=1) -> TensorDataset
```

2-D classification problems designed for quick experiments:

| Name | Shape |
|------|-------|
| `"simple"` | separable gaussian blobs |
| `"diag"` | diagonal stripes |
| `"split"` | split advice region |
| `"xor"` | XOR corners |
| `"circle"` | concentric circles |
| `"spiral"` | two interleaved arms |

```python
ds = make_synthetic("spiral", n=300, seed=7)
x, y = ds.to_batch()
```

### synthetic_classification / make_spiral

```python
def synthetic_classification(name, n=100, seed=1) -> Graph
def make_spiral(N) -> List[Tuple[float, float]]
```

Lower-level entry points returning the raw coordinate/label `Graph` and the
spiral point list respectively.