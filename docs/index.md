# Torchlight

**A lightweight, from-scratch deep learning framework** — with the same high-level
ergonomics as PyTorch (`tensor`, `nn`, `optim`, `data`), a tiny reverse-mode
autograd engine, and a numpy-backed CPU backend with optional numba and CUDA
accelerators. Built for learning and for small-scale experiments, and for
reading **every line** of what makes deep learning tick.

```python
import torchlight as tl
from torchlight.nn import Linear

x = tl.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad.to_numpy())          # [2.0, 4.0, 6.0]

model = Linear(2, 1)              # high-level API feels like torch
```

## System design at a glance

Everything above the kernel sits on a deliberately small stack. One diagram
for the whole system:

```mermaid
flowchart TB
    subgraph User["Your code"]
        A1["tensor ops / autograd"]
        A2["nn modules"]
        A3["optimizers"]
        A4["data loaders"]
        A5["jit trace / persistence"]
    end

    subgraph API["torchlight public API"]
        B["Tensor / factories"]
        C["Module / Parameter tree"]
        D["Optimizer / Scheduler"]
        E["Dataset / DataLoader"]
        F["trace / GraphTape / save-load"]
    end

    subgraph Core["Core engine"]
        G["Autograd: Function · Context · History"]
        H["TensorData: flat float32 storage + shape/strides"]
    end

    subgraph Be["Backend layer"]
        I["primitives: map / zip / reduce / matmul + fused softmax · layernorm"]
        J["CPUBackend — numpy"]
        K["NumbaBackend — numba CPU-JIT"]
        L["CudaBackend — numba-cuda or ctypes '.so'"]
    end

    subgraph HW["Hardware"]
        M["CPU"]
        N["NVIDIA GPU"]
    end

    A1 --> B
    A2 --> C
    A3 --> D
    A4 --> E
    A5 --> F
    B --> G
    C --> G
    E --> B
    G --> H
    H --> I
    I --> J
    I --> K
    I --> L
    J --> M
    K --> M
    L --> N
```

The bet of the design: **reduce everything to tensors, and compute everything
through a handful of primitives**. High-level layers (`nn`, `optim`, `data`,
`jit`) are pure bookkeeping on top; the math never leaves the primitive layer.

## 1. Tensors: numbers, shapes, and storage

A `Tensor` is a small object; the real data lives in a `TensorData` — a flat
one-dimensional `float32` numpy buffer described by shape/strides. Views,
broadcasts and transposes change *strides*, never the buffer:

```mermaid
flowchart LR
    T["Tensor\nshape · strides · requires_grad · history\n(one node in the autograd graph)"] --> TD["TensorData\nflat float32 storage"]
    TD --> V["numpy_view() — strided numpy view, zero copy"]
    V --> U["ufunc / numba / cuda kernels"]
    TD --> B["view / permute / broadcast_to\nshape + strides only — no data copied"]
```

- broadcasting = stride-0 dimensions; nothing is materialised,
- `view`/`reshape` reuse storage when contiguous, otherwise materialise,
- every factory accepts `requires_grad=True` and a `device=`.

## 2. The autograd engine

Reverse-mode automatic differentiation over a **dynamic** graph. Each forward
op is a `Function` that records a `History` node (its `Function` subclass, the
saved input tensors, and a `Context` for forward-side values) on its outputs;
calling `backward()` on the loss walks the graph in reverse topological order,
invoking each `Function.backward(ctx, grad_out)` to produce input gradients:

```mermaid
sequenceDiagram
    participant U as user code
    participant T as Tensor
    participant F as Function.apply
    participant BE as backend (numpy / numba / cuda)
    participant H as History

    U->>T: out = a + b  (inputs require grad)
    T->>F: apply(ctx, a, b)
    F->>BE: forward via zip / map / reduce / matmul
    BE-->>F: results
    F->>T: build out Tensor
    F->>H: record (fn, saved inputs, ctx) — the graph node
    U->>T: loss.backward()
    T->>H: topological_sort(loss)
    loop reverse topological order
        H->>F: backward(ctx, grad_output)
        F->>T: gradient wrt each saved input
    end
    T-->>U: a.grad, b.grad accumulated and ready
```

The whole engine is `src/torchlight/autograd/` (`autodiff.py` +
`functions.py` + `convolutions.py` + `indexing.py`) and is validated against
finite-difference gradchecks in the test suite. A scalar twin
(`tl.Scalar`, `tl.ScalarFunction`, `tl.derivative_check`) teaches the same
math on plain numbers.

## 3. Modules and parameters

`nn.Module` intercepts attribute assignment: a `Parameter` lands in
`_parameters`, a child `Module` in `_modules`, anything else in `__dict__`.
`parameters()`/`named_parameters()` then walk that tree recursively, so Adam
and the persistence layer can see every trainable tensor with zero extra
registrations:

```mermaid
flowchart TB
    Model["class SentimentNet(nn.Module)"] --> Emb["Embedding(V, 128) — weight: Parameter"]
    Model --> Seq["nn.Sequential(...)"]
    Seq --> L1["Linear(128 → 64) — weight · bias"]
    L1 --> R["ReLU"]
    R --> Drop["Dropout(0.3)"]
    Drop --> L2["Linear(64 → 1) — weight · bias"]
    Model -. "named_parameters() recursively walks _modules" .-> Params["4 parameter tensors"]
    Params --> Opt["SGD / Adam / AdamW"]
    Params --> Save["save_state → out/*.npz"]
```

Layers shipped: `Linear`, `Embedding`, `Conv1d/2d`, norm (`LayerNorm`,
`BatchNorm1d`), activations (`ReLU`, `Sigmoid`, `Tanh`, `GELU`, ...), pooling,
`Dropout`, plus losses and `F.masked_*`/`F.softmax_loss` helpers.

## 4. Optimizers and the training loop

Every optimizer gets one job — turn `param.grad` into a `param`
update — and one protocol: `parameters()`, `zero_grad()`, `step()`
(schedulers adjust the learning rate between epochs). The canonical loop:

```mermaid
flowchart TD
    A["Dataset / DataLoader\n(shuffle → batches of (x, y))"] --> B["logits = model(x_batch)"]
    B --> C["loss = F.cross_entropy(logits, y)"]
    C --> D["loss.backward()"]
    D --> E["every Parameter.grad filled by autograd"]
    E --> F["optimizer.step()\nSGD · Adam · AdamW rules"]
    F --> G["optimizer.zero_grad()"]
    G --> A
```

| Optimizer                     | Update rule (per parameter)                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| `SGD (+momentum, nesterov)` | $p \leftarrow p - \eta \cdot m$                                      |
| `Adam`                      | $p \leftarrow p - \eta\,\dfrac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}$ |
| `AdamW`                     | Adam + decoupled weight decay$p\leftarrow p - \eta\lambda p$         |

LR schedulers (`StepLR`, `MultiStepLR`, `ExponentialLR`,
`CosineAnnealingLR`) shrink $\eta$ between epochs.

## 5. The data pipeline

```mermaid
flowchart LR
    Src["raw data (numpy / files)"] --> TS["TensorDataset(x_tensor, y_tensor)"]
    TS --> DL["DataLoader(batch_size, shuffle=True, seed)"]
    DL --> R["RandomSampler — permuted indices"]
    R --> B["BatchSampler — contiguous batch slices"]
    B --> T["(xb, yb) torchlight Tensors, ready for model(xb)"]
    DL --> Seq["SequentialSampler — stable order for eval"]
```

Plus a synthetic-problem zoo (`make_synthetic`: `Simple`, `Diag`, `Split`,
`Xor`, `Circle`, `Spiral`) for smoke-testing a model before you feed it real
data.

## 6. Backends: the whole hardware layer is six primitives

All compute passes through **map / zip / reduce / matmul** (+ fused
softmax/LayerNorm kernels). That contract is small enough to reimplement in
numpy, numba and CUDA, so device parity comes from one interface. The device
argument is resolved through one funnel:

```mermaid
flowchart LR
    Dev["device= argument\n('cpu' / 'cuda' / Device / None)"] --> Res["resolve_device()"]
    Env["TORCHLIGHT_DEVICE env var\n(default 'cpu')"] --> Res
    Res --> CPUC["device('cpu:0')\nCPUBackend — numpy (always available)"]
    Res --> CUD["device('cuda:0')"]
    CUD --> Lazy["lazily import .cuda (no numba cost at import)"]
    Lazy --> NB{"numba installed?"}
    NB -- "no" --> E1["RuntimeError: install pip .[cuda]"]
    NB -- "yes" --> Run{"CUDA driver / GPU?"}
    Run -- "no" --> E2["RuntimeError: no driver found"]
    Run -- "yes" --> K["numba-cuda kernels · or ctypes '.so' from cuda_kernels/"]
```

Guarantee: `import torchlight` is always fast and pure-numpy. Acceleration is
pulled in only when a kernel is actually requested.

## 7. JIT-ish tracing and persistence

`jit.trace` records a forward pass as an *op tape* (`GraphTape` of
`OpRecord`s) using the autograd hook — then you can count ops or replay the
program on new inputs without rebuilding any graph:

```mermaid
flowchart LR
    M["model.forward(sample_batch)"] --> T["trace() — op callback hooked into autograd"]
    T --> R["OpRecord per op (name, shapes)"]
    R --> G["GraphTape — linearized op program"]
    G --> C["count_ops() → {op: count}"]
    G --> E["exec_tape() → replay on fresh inputs, no graph"]
```

Persistence reuses the same parameter walk as training: `state_dict` →
`.npz` checkpoints (`save_state` / `load_state`) or whole-model pickling
(`save_model` / `load_model`). Every project in `projects/` saves
`out/<name>_model.npz` + `meta.json`, and each ships a `post.py` that reloads,
evaluates and plots.

## Features

- **Autograd engine** — reverse-mode auto-diff over a dynamic computation graph
  (`Function`, `Context`, `History`) with finite-difference gradchecks.
- **Tensor core** — N-dimensional, CPU-first tensors backed by numpy: strided
  views, broadcasting, and vectorised kernels (no element-level Python loops).
- **Neural network module** — `Module`/`Parameter` tree, `Linear`, `Sequential`,
  activations, dropout, normalization, pooling, and functional losses.
- **Optimizers** — `SGD` (momentum/nesterov), `Adam`, and `AdamW` + LR schedulers.
- **Data pipeline** — `Dataset` / `TensorDataset` / `DataLoader` with batching
  and shuffling, plus a synthetic-problem zoo (`Simple`, `Split`, `Xor`,
  `Circle`, `Spiral`, `Diag`).
- **JIT-ish tooling** — trace a forward pass as an op tape, count ops, and
  replay it without rebuilding the autograd graph.
- **Persistence** — state dicts (`.npz`) and whole-model pickling.
- **Swap-able backends** — the whole hardware layer is four primitives + fused
  ops: numpy CPU out of the box, numba-cuda for NVIDIA GPUs as an optional install.

## Install

```bash
pip install -e .                # CPU backend (numpy) — works anywhere
pip install -e ".[cuda]"        # NVIDIA GPU backend (numba-cuda) — add GPU support
pytest                          # run the test suite (requires test extra)
```

No `PYTHONPATH` tricks needed for the docs build or tests — `src/` layout is
handled via `[tool.pytest.ini_options].pythonpath`.

## Quick start

```python
import torchlight as tl
from torchlight.nn import Sequential, Linear, ReLU
from torchlight.nn.functional import cross_entropy
from torchlight.optim import SGD

# Tiny MLP on synthetic classification data
from torchlight.data import make_synthetic

train = make_synthetic("xor", n=200)
x, y = train.to_batch()

model = Sequential(Linear(2, 16), ReLU(), Linear(16, 2))
opt = SGD(model.parameters(), lr=0.5)

for _ in range(20):
    opt.zero_grad()
    out = model(x)
    loss = cross_entropy(out, y)
    loss.backward()
    opt.step()
    print(f"loss = {float(loss.to_numpy().ravel()[0]):.4f}")
```

## Documentation map

**Home · System design** (this page) explains the layers; the tutorials take
them one by one:

- **Tutorials** — [Tensors](./tutorials/01-tensors.md), [Autograd](./tutorials/02-autograd.md),
  [Neural networks](./tutorials/03-neural-networks.md), [Data &amp; optimizers](./tutorials/04-data-and-optimizers.md),
  [Backends](./tutorials/05-backends.md), [JIT &amp; persistence](./tutorials/06-jit-and-persistence.md),
  [Real-world datasets](./tutorials/07-real-world-datasets.md).
- **Projects** (end-to-end, first-principles maths): [Sentiment classifier](./projects/sentiment-classifier.md),
  [Machine translation](./projects/machine-translation.md), [Transformer from scratch](./projects/transformers-from-scratch.md).
- **Examples**: [MLP on Breast Cancer](./examples/sklearn-classifier.md), [CNN on digits](./examples/sklearn-digits-conv.md),
  [Autoencoder](./examples/sklearn-autoencoder.md), [CUDA kernels](./examples/cuda.md).
- **API reference** — [core](./api/core.md), [tensor](./api/tensor.md),
  [autograd](./api/autograd.md), [nn](./api/nn.md), [optim](./api/optim.md),
  [data](./api/data.md), [jit](./api/jit.md), [backends](./api/backends.md), [utils](./api/utils.md).

## Project layout

```
torchlight/
├── src/torchlight/       # installable package
│   ├── core/             # tensor_data (strides/broadcast), device, scalar ops
│   ├── backends/         # CPU / numba / cuda backends: map / zip / reduce / matmul
│   ├── autograd/         # reverse-mode autodiff: autodiff.py, functions.py
│   ├── tensor/           # Tensor object + factories (tensor, zeros, randn, ...)
│   ├── nn/               # Module, Parameter, Linear, Embedding, activations, losses, norm
│   ├── optim/            # SGD, Adam, AdamW + LR schedulers
│   ├── data/             # Dataset, DataLoader, synthetic problems
│   ├── utils/            # state_dict, save/load models
│   └── jit/              # trace -> GraphTape -> replay / op counts
├── tests/                # pytest suite (gradchecks, nn, optim, data, jit)
├── examples/             # intro.py, persistence.py, sklearn demos, CUDA self-tests
├── projects/             # end-to-end experiments on real data (see Projects above)
├── benchmarks/           # perf and comparison benchmarks
└── docs/                 # this documentation
```
