# Torchlight

A lightweight, from-scratch deep learning framework inspired by PyTorch — built for learning and for small-scale LLM experiments.

## Features

- **Autograd engine** — reverse-mode auto-diff over a dynamic computation graph (`Context`, `History`, [Function](./src/torchlight/autograd/functions.py)) with numeric gradchecks
- **Tensor core** — N-dimensional, CPU-first tensors backed by numpy; strided views, broadcasting, and vectorised kernels (no element-level Python loops)
- **Advanced indexing** — differentiable Python-style slicing (`x[:, 1::-1]`, `None` axes, ellipsis), `unsqueeze`, `index_select`, and `cat` / `stack` / `split` / `chunk`
- **Neural network module** — `Module`/`Parameter` tree, `Linear`, `Sequential`, `Embedding`, activations, dropout, LayerNorm/BatchNorm, pooling, and functional losses
- **Sequence utilities** — masked pooling (`F.masked_sum/mean/max`) for variable-length inputs and `nn.utils.clip_grad_norm_`
- **Optimizers & schedulers** — SGD (momentum/nesterov), Adam, AdamW (decoupled weight decay), plus `StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`
- **Data pipeline** — `Dataset` / `TensorDataset` / `DataLoader` with shuffling and batching, plus a synthetic-problem zoo (`Simple/Diag/Split/Xor/Circle/Spiral`)
- **JIT-ish tooling** — [op tracing](./src/torchlight/jit/trace.py): record a forward pass as an op tape, count ops, and replay it without rebuilding the autograd graph
- **Persistence** — state dicts (`.npz`) and whole-model pickling in [`utils/serialization.py`](./src/torchlight/utils/serialization.py)
- **GPU backends** — numpy on CPU, numba-CPU/GPU, and CUDA-C kernels (compiled to `.so`, loaded via ctypes)

## Getting Started

```bash
pip install -e .            # CPU backend (numpy) — works anywhere, nothing extra needed
pip install -e ".[cuda]"    # NVIDIA GPU backend (numba-cuda) — add GPU support
pytest                      # 250+ tests, incl. finite-difference gradchecks
```

There is no separate CPU-JIT install: numpy covers every CPU. The only optional
install that unlocks a new device is `cuda` (it pulls in numba-cuda).

```python
import torchlight as tl
from torchlight.nn import Linear, ReLU
from torchlight.optim import Adam

x = tl.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad.to_numpy())  # [2.0, 4.0, 6.0]

model = Linear(2, 2)      # high-level API feels like torch
```

Run the end-to-end demo: `PYTHONPATH=src python examples/intro.py`.

### Examples

| Example                                                        | What it shows                                                          | Run                                                       |
| -------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------- |
| [`intro.py`](./examples/intro.py)                             | tensor ops, autograd, a tiny MLP                                       | `PYTHONPATH=src python examples/intro.py`               |
| [`persistence.py`](./examples/persistence.py)                 | save/load state dicts and whole models                                 | `PYTHONPATH=src python examples/persistence.py`         |
| [`sklearn_classifier.py`](./examples/sklearn_classifier.py)   | MLP on Breast Cancer (BatchNorm + Dropout + cosine LR + grad clipping) | `PYTHONPATH=src python examples/sklearn_classifier.py`  |
| [`sklearn_digits_conv.py`](./examples/sklearn_digits_conv.py) | CNN on handwritten digits (Conv2d + MaxPool2d)                         | `PYTHONPATH=src python examples/sklearn_digits_conv.py` |
| [`sklearn_autoencoder.py`](./examples/sklearn_autoencoder.py) | class-conditioned autoencoder on digits (Embedding +`cat`)           | `PYTHONPATH=src python examples/sklearn_autoencoder.py` |

The `sklearn_*` examples use real datasets from
[scikit-learn](https://scikit-learn.org) (`pip install scikit-learn`).

### Documentation

The full docs (tutorials + projects + examples + API reference) are built with
`mkdocs` (maths render via `$$...$$` + MathJax):

```bash
pip install mkdocs mkdocs-material
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site in ./site
```

## Directory Layout

```
torchlight/
├── src/torchlight/       # installable package
│   ├── core/             # tensor_data (strides/broadcast), device, scalar ops
│   ├── backends/         # CPU / numba / cuda backends: map / zip / reduce / matmul
│   ├── autograd/         # reverse-mode autodiff: autodiff.py, functions.py, convolutions.py, indexing.py
│   ├── tensor/           # Tensor object + factories (tensor, zeros, randn, ...)
│   ├── nn/               # Module, Parameter, Linear, Embedding, activations, losses, norm, utils
│   ├── optim/            # SGD, Adam, AdamW, LR schedulers
│   ├── data/             # Dataset, DataLoader, synthetic problems
│   ├── utils/            # state_dict, save/load models
│   ├── jit/              # trace -> GraphTape -> replay / op counts
│   └── cuda_kernels/     # CUDA-C sources (softmax/layer-norm kernels, .so via nvcc)
├── tests/                # pytest suite (gradchecks, nn, optim, data, jit)
├── examples/             # intro, persistence + sklearn-classifier/CNN/autoencoder demos
├── projects/             # end-to-end experiments on real data
│   ├── sentiment-classifier/         # NLP: embedding + MLP sentiment (UCI reviews)
│   ├── machine-translation/          # NLP: GRU seq2seq + attention (EN->FR Tatoeba)
│   └── transformers-from-scratch/    # LLM: miniature GPT on tiny Shakespeare
├── benchmarks/           # perf and comparison benchmarks
├── docs/                 # mkdocs site: tutorials + projects + examples + API
```

## Roadmap

- [X] Tensor core with broadcasting and basic ops
- [X] Autograd engine (backprop through graph)
- [X] nn module hierarchy (Linear, activations, Sequential, norm, dropout)
- [X] Optimizers (SGD, Adam, AdamW)
- [X] DataLoader with batching and shuffling
- [ ] Mini-LLM training example (attention layer, embeddings, sampler)

## License

MIT
