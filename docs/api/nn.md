# API Reference: nn

`torchlight.nn` provides PyTorch-like building blocks: modules, layers,
activations, normalization, pooling, and losses.

## Modules

### Module

```python
class Module()
```

Base class for all modules. Child modules and `Parameter`s register
automatically on attribute assignment.

| Method                         | Description                         |
| ------------------------------ | ----------------------------------- |
| `forward(*args, **kwargs)`   | override in subclasses              |
| `__call__(...)`              | calls`forward`                    |
| `parameters()`               | all parameters (recursive)          |
| `named_parameters()`         | `(name, Parameter)` pairs, dotted |
| `modules()`                  | direct child modules                |
| `named_modules()`            | `(name, Module)` pairs, recursive |
| `add_module(name, module)`   | register a child module             |
| `add_parameter(name, param)` | register a parameter                |
| `train()`                    | set training mode (recursive)       |
| `eval()`                     | set evaluation mode (recursive)     |

### Parameter

```python
class Parameter(data, name=None, requires_grad=True)
```

A trainable wrapper around a `Tensor`.

| Attribute         | Meaning                                        |
| ----------------- | ---------------------------------------------- |
| `value`         | the underlying tensor                          |
| `data`          | alias for`value`                             |
| `name`          | optional name                                  |
| `update(value)` | rebind to a detached leaf (used by optimizers) |

### Sequential

```python
class Sequential(*modules: Module)
```

A container that applies modules in order.

### Linear

```python
class Linear(in_features, out_features, bias=True)
```

Fully-connected layer: `x @ W^T + b`, weight shape `(out, in)`, He-initialised.

### Embedding

```python
class Embedding(num_embeddings, embedding_dim, padding_idx=None)
```

Looks up rows of a learnable `(num_embeddings, embedding_dim)` weight matrix
by integer index.  Weights are drawn from `N(0, 1)`.

| Argument           | Meaning                                    |
| ------------------ | ------------------------------------------ |
| `num_embeddings`  | size of the vocabulary                     |
| `embedding_dim`   | dimension of each embedding vector         |
| `padding_idx`     | if set, that row stays zero and receives no gradient |

```python
emb = Embedding(1000, 64, padding_idx=0)
ids = tl.tensor([1, 42, 7])
vec = emb(ids)                # (3, 64)
```

### Dropout

```python
class Dropout(p=0.5)
```

Inverted dropout; active only while `training=True`.

## Activations

All activations are parameter-free.

| Class                              | Formula                       |
| ---------------------------------- | ----------------------------- |
| `ReLU()`                         | `max(x, 0)`                 |
| `LeakyReLU(negative_slope=0.01)` | `x if x>0 else slope*x`     |
| `Sigmoid()`                      | stable logistic               |
| `Tanh()`                         | hyperbolic tangent            |
| `Softmax(dim=-1)`                | normalized exponentials       |
| `LogSoftmax(dim=-1)`             | log-domain stable log-softmax |
| `GELU()`                         | Gaussian Error Linear Unit    |

## Normalization

### BatchNorm1d

```python
class BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True)
```

### LayerNorm

```python
class LayerNorm(normalized_shape, eps=1e-5, affine=True)
```

### LayerNorm1d

```python
class LayerNorm1d(num_features, eps=1e-5, affine=True)
```

LayerNorm applied to the last dimension; convenience wrapper around
`LayerNorm((num_features,))`.

## Pooling

```python
class MaxPool2d(kernel: int | (kh, kw))
class AvgPool2d(kernel: int | (kh, kw))
```

Pool a `(B, C, H, W)` tensor with a kernel window.

## Masked pooling

For variable-length sequences.  `mask` has the *leading* dims of `input`
(e.g. `(B, L)` over `(B, L, E)`) and marks valid positions with `1.0`.
Masked positions contribute zero to the sum, are ignored in the mean, and
are parked at `−1e9` before the max.

| Function            | Signature                     | Notes                                          |
| ------------------- | ----------------------------- | ---------------------------------------------- |
| `F.masked_sum`    | `(input, mask, dim)`        | fully-masked rows → 0                           |
| `F.masked_mean`   | `(input, mask, dim)`        | fully-masked rows → 0 (not NaN)                 |
| `F.masked_max`    | `(input, mask, dim)`        | reduces the masked axis (axis size 1 dropped) |

```python
from torchlight.nn import functional as F

x    = tl.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mask = tl.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

F.masked_mean(x, mask, 1)   # [1.5, 5.5]  — mean of valid positions
F.masked_max(x, mask, 1)    # [2.0, 6.0]
```

## Gradient clipping

```python
from torchlight.nn.utils import clip_grad_norm_

clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
```

Clips the total gradient norm to `max_norm` (in-place).  Returns the
**pre-clip** total norm as a float.  Supports `norm_type=float("inf")`.

## Losses

| Class                        | Computation                                   |
| ---------------------------- | --------------------------------------------- |
| `MSELoss()`                | `mean((pred - target)^2)`                   |
| `L1Loss()`                 | `mean(abs(pred - target))`                  |
| `CrossEntropyLoss(dim=-1)` | gathered negative log-softmax (index targets) |
| `BCELoss()`                | log-stable BCE on probabilities               |
| `BCEWithLogitsLoss()`      | BCE on raw logits (softplus form)             |

## Functional API

Stateless versions of everything live in `torchlight.nn.functional`:

```python
from torchlight.nn.functional import (
    relu, leaky_relu, sigmoid, tanh, softmax, log_softmax, dropout,
    mse_loss, l1_loss, binary_cross_entropy, binary_cross_entropy_with_logits,
    cross_entropy, nll_loss,
    layer_norm, avg_pool2d, max_pool2d,
    embedding, masked_sum, masked_mean, masked_max,
)
```

| Function                                        | Signature                          | Notes                                  |
| ----------------------------------------------- | ---------------------------------- | -------------------------------------- |
| `relu(x)`                                     | elementwise ReLU                   |                                        |
| `sigmoid(x)`                                  | stable logistic                    |                                        |
| `dropout(x, p=0.5, training=True)`            | inverted dropout                   |                                        |
| `embedding(input, weight, padding_idx=None)`   | `(input, E)` or `(*, E)`         | `input` is an integer tensor           |
| `mse_loss(x, y, reduction="mean")`            | mean squared error                 |                                        |
| `cross_entropy(x, target, dim=-1)`            | index-target CE                    |                                        |
| `binary_cross_entropy(x, target)`             | BCE on probabilities               |                                        |
| `binary_cross_entropy_with_logits(x, target)` | BCE on logits                      |                                        |
| `masked_sum(x, mask, dim)`                    | sum valid positions                | axis dropped                           |
| `masked_mean(x, mask, dim)`                   | mean valid positions               | axis dropped; all-masked → 0           |
| `masked_max(x, mask, dim)`                    | max valid positions                | axis dropped; all-masked → −1e9        |

## Init (initializers)

```python
from torchlight.nn import init

init.uniform_(t, low=0.0, high=1.0)     # in-place U(low, high)
init.normal_(t, mean=0.0, std=1.0)      # in-place N(mean, std)
init.xavier_uniform_(t, gain=1.0)
init.xavier_normal_(t, gain=1.0)
init.kaiming_uniform_(t, a=0.0)
init.kaiming_normal_(t, a=0.0)
init.zeros_(t); init.ones_(t)
```
