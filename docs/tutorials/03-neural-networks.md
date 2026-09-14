# Tutorial 3: Neural Networks

`torchlight.nn` gives you a PyTorch-like module system: `Module`, `Parameter`,
layers, activations, losses — all built on top of the Tensor + autograd engine.

## Module and Parameter

Every trainable unit inherits from `Module`:

```python
from torchlight.nn import Module, Parameter

class Simple(Module):
    def __init__(self):
        super().__init__()
        self.w = Parameter(tl.randn(3))   # trainable
        self.b = Parameter(tl.zeros(1))   # trainable

    def forward(self, x):
        return x * self.w + self.b

model = Simple()
model.parameters()            # [Parameter(w), Parameter(b)]
```

Child `Module`s are also registered automatically:

```python
class Net(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(4, 16)
        self.layer2 = Linear(16, 1)

    def forward(self, x):
        return self.layer2(relu(self.layer1(x)))
```

## Linear layer

`Linear(in_features, out_features)` computes `x @ W^T + b` with He-initialised
weights:

```python
from torchlight.nn import Linear

layer = Linear(4, 3)
x = tl.tensor([[1.0, 2.0, 3.0, 4.0]])
layer(x).shape                # (1, 3)
```

## Activations

All activations have no learnable parameters:

```python
from torchlight.nn import ReLU, Sigmoid, Tanh, LeakyReLU, Softmax, LogSoftmax
from torchlight.nn.functional import relu, sigmoid, tanh, softmax

# Module form (use inside Sequential)
modules: ReLU(), Sigmoid(), Tanh(), LeakyReLU(0.01), Softmax(dim=-1)

# Functional form (use anywhere)
relu(x); sigmoid(x); tanh(x); softmax(x, dim=-1)
```

## Building a model with Sequential

`Sequential` chains modules in order:

```python
from torchlight.nn import Sequential, Linear, ReLU, Dropout

model = Sequential(
    Linear(10, 32),
    ReLU(),
    Dropout(0.5),
    Linear(32, 1),
    Sigmoid(),
)
model(tl.randn((4, 10))).shape    # (4, 1)
```

## Normalization

```python
from torchlight.nn import BatchNorm1d, LayerNorm

bn   = BatchNorm1d(32)
ln   = LayerNorm([16, 32])
```

## Pooling

```python
from torchlight.nn import MaxPool2d, AvgPool2d

pool = MaxPool2d((2, 2))
```

## Losses

All losses follow the pattern `loss(output, target) -> scalar tensor`:

```python
from torchlight.nn import MSELoss, L1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torchlight.nn.functional import (
    mse_loss, l1_loss, cross_entropy,
    binary_cross_entropy, binary_cross_entropy_with_logits,
)

# For multi-class (logits, integer targets)
loss = CrossEntropyLoss()(logits, target)   # shape (1,)

# For binary (probabilities in [0,1])
loss = BCELoss()(prob, target)

# For binary (raw logits — more stable)
loss = BCEWithLogitsLoss()(logit, target)
```

## Training loop pattern

The standard pattern, identical to PyTorch:

```python
from torchlight.optim import Adam

model = Sequential(Linear(2, 16), ReLU(), Linear(16, 1))
opt = Adam(model.parameters(), lr=0.01)

for x, y in loader:                   # x, y are Tensor batches
    opt.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    opt.step()                         # gradients flow, params update
```

## Train / eval mode

Dropout and BatchNorm respect `model.train()` vs. `model.eval()`:

```python
model.train()    # dropout active, BN uses batch stats
model.eval()     # dropout off, BN uses running stats
```

## Naming parameters

Named parameters are exposed for serialization:

```python
for name, p in model.named_parameters():
    print(name, p.value.shape)
# linear.weight (3, 2)
# linear.bias   (3,)
```

## Next

See [Data and optimizers](./04-data-and-optimizers.md) for `DataLoader`,
synthetic datasets, and optimizer details.
