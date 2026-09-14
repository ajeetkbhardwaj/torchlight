# Tutorial 4: Data and Optimizers

Torchlight's `data` module gives you `Dataset`, `TensorDataset`, and
`DataLoader` — all implemented in pure Python/numpy on top of the Tensor API.

## Dataset

The abstract base class: implement `__len__` and `__getitem__(idx)`.

```python
from torchlight.data import Dataset

class MyData(Dataset):
    def __init__(self, X, y):
        self.X = tl.tensor(X)
        self.y = tl.tensor(y)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## TensorDataset

A ready-made dataset from parallel arrays:

```python
from torchlight.data import TensorDataset

X = tl.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = tl.tensor([0.0, 1.0, 1.0])
ds = TensorDataset(X, y)

len(ds)          # 3
ds[0]            # (tensor([1., 2.]), tensor([0.])) — single element as tensors
ds.to_batch()    # (X, y) as full tensors — handy for small datasets
```

## DataLoader

Wraps a dataset with batching, shuffling, and dropping:

```python
from torchlight.data import DataLoader

loader = DataLoader(ds, batch_size=2, shuffle=True, drop_last=False)

for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)   # (2, 2) (2,) for full batch; (1, 2) for last
```

Key arguments:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `batch_size` | 1 | elements per batch |
| `shuffle` | `False` | randomise order each epoch |
| `drop_last` | `False` | discard last incomplete batch |
| `seed` | `None` | RNG seed for reproducibility |

## Synthetic datasets

Pre-built problems for quick experiments:

```python
from torchlight.data import make_synthetic

for name in ["simple", "diag", "split", "xor", "circle", "spiral"]:
    ds = make_synthetic(name, n=200)
    x, y = ds.to_batch()
    print(name, x.shape, y.shape)
```

## Optimizers

All optimizers follow the same interface: `zero_grad()` → forward →
`loss.backward()` → `step()`.

### SGD

Basic stochastic gradient descent (with optional momentum):

```python
from torchlight.optim import SGD

opt = SGD(model.parameters(), lr=0.1)                # vanilla
opt = SGD(model.parameters(), lr=0.1, momentum=0.9)  # with momentum
opt = SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)  # Nesterov
```

### Adam

Adaptive learning rate with bias correction:

```python
from torchlight.optim import Adam

opt = Adam(model.parameters(), lr=0.001)                    # default betas
opt = Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # with L2 decay
```

### AdamW

Decoupled weight decay (Loshchilov & Hutter 2019) — recommended for larger
models:

```python
from torchlight.optim import AdamW

opt = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## Full training example

```python
import torchlight as tl
from torchlight.nn import Sequential, Linear, ReLU
from torchlight.nn.functional import cross_entropy
from torchlight.optim import Adam
from torchlight.data import make_synthetic, DataLoader

train_ds = make_synthetic("xor", n=200)
loader = DataLoader(train_ds, batch_size=32, shuffle=True)
model = Sequential(Linear(2, 32), ReLU(), Linear(32, 2))
opt = Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    epoch_loss = 0.0
    for bx, by in loader:
        opt.zero_grad()
        loss = cross_entropy(model(bx), by)
        loss.backward()
        opt.step()
        epoch_loss += float(loss.to_numpy().ravel()[0])
    if epoch % 10 == 0:
        print(f"epoch {epoch}: loss = {epoch_loss / len(loader):.4f}")
```

## Next

Learn about swapping hardware with [Backends](./05-backends.md).