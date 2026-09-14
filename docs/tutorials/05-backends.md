# Tutorial 5: Backends

The entire hardware layer in Torchlight is **six primitive functions** —
`map`, `zip`, `reduce`, and `matmul` — wrapped into a backend class.
Swap the backend and autograd, `nn`, and optimizers keep working unchanged.

## CPU (numpy) — the default

```python
from torchlight.backends import CPUBackend

backend = CPUBackend()
x = tl.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sigmoid().sum()
y.backward()
# Everything runs through numpy vectorised ufuncs — no Python loops.
```

**No extra install needed.** Numpy covers every CPU; there is deliberately no
separate CPU-JIT install.

## CUDA (NVIDIA GPU)

numba-cuda kernels that transfer data to the GPU and back per-operation:

```python
x = tl.tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")
y = (x ** 2).sum()
y.backward()           # kernels run on the GPU, result pulled back to CPU
```

Requires an NVIDIA GPU + driver, and the `cuda` extra install (which pulls
in numba-cuda automatically):

```bash
pip install "torchlight[cuda]"
```

!!! note
    The CudaBackend is implemented and verified via the `leetgpu` simulator
    but has not been tested on real NVIDIA hardware.

## Fused ops (attention-softmax / layer-norm)

Beyond the four primitives, every backend also exposes three **fused kernel
primitives** used by the attention-softmax and layer-norm ops:

| Primitive | Description |
|-----------|-------------|
| `attn_softmax_fw(out, inp, mask, rows, cols)` | `softmax(inp + mask)` over last axis |
| `attn_softmax_bw(out, grad, soft, rows, cols)` | softmax backward |
| `layernorm_fw(out, inp, gamma, beta, rows, cols)` | layer norm forward |
| `layernorm_bw(dinp, dgamma, dbeta, grad, inp, gamma, rows, cols)` | layer norm backward |

The CPU backend implements these in numpy.  The numba backend JIT-compiles
identical loops, and `CudaBackend` launches `numba.cuda.jit` mirror kernels
with shared-memory tree reductions.

**When the real CUDA C kernels are compiled to `.so`**, `CudaBackend` loads
them via ctypes (the same mechanism MiniTorch uses) and routes all fused ops
through the compiled kernel launchers instead:

```bash
# Compile on a CUDA host (same commands as MiniTorch)
nvcc -O3 --shared -o softmax_kernel.so \
    softmax_kernel.cu -Xcompiler -fPIC -std=c++14

nvcc -O3 --shared -o layernorm_kernel.so \
    layernorm_kernel.cu -Xcompiler -fPIC -std=c++14
```

Place them in `torchlight/cuda_kernels/` or point at them via:

```bash
export TORCHLIGHT_CUDA_KERNEL_DIR=/path/to/compiled/kernels
```

!!! note
    The CUDA C kernels are validated locally by the LeetGPU simulator:

    ```bash
    leetgpu run examples/cuda/fused_kernels_test.cu \
                src/torchlight/cuda_kernels/fused_kernels.h \
                src/torchlight/cuda_kernels/softmax_kernel.cu \
                src/torchlight/cuda_kernels/layernorm_kernel.cu
    ```

## Picking a backend

The `device=` kwarg on any tensor factory controls which backend handles the
op:

```python
x = tl.zeros((10, 10), device="cpu")    # numpy
x = tl.zeros((10, 10), device="cuda")   # numba-cuda on GPU
```

Or set it globally via an environment variable:

```bash
export TORCHLIGHT_DEVICE=cuda
python train.py
```

```python
import os
os.environ["TORCHLIGHT_DEVICE"] = "cuda"
x = tl.tensor([1.0, 2.0])   # automatically goes to the cuda backend
```

## Programmatic backend selection

```python
from torchlight.backends import get_backend

backend = get_backend("cuda")        # returns a CudaBackend instance
backend = get_backend("cpu")         # returns the default CPUBackend singleton
backend = get_backend(None)          # honours TORCHLIGHT_DEVICE or defaults to cpu
```

!!! note
    On a machine without the `cuda` extra installed, `get_backend("cuda")`
    (and `device="cuda"`) raises a clear error explaining what to install.

## How it works internally

The backends expose:

| Primitive | Signature | Description |
|-----------|-----------|-------------|
| `map(fn)` | `fn(TensorData) -> TensorData` | unary elementwise |
| `zip(fn)` | `fn(TensorData, TensorData) -> TensorData` | binary elementwise (broadcasts) |
| `reduce(ufunc, start)` | `TensorData -> TensorData` | reduce one dimension |
| `matmul(a, b)` | `(TensorData, TensorData) -> TensorData` | (batched) matrix multiply |

`CPUBackend` implements each with a numpy ufunc; `CudaBackend` looks up the
op in a table of pre-compiled numba-cuda kernels (keyed by a function ID),
or falls back to the CPU numpy path if no kernel exists.

> Internals: `NumbaBackend` is the CPU twin of `CudaBackend` — it runs the
> same-style kernels JIT-compiled on the CPU. It is used to develop and
> validate the GPU kernels on machines without an NVIDIA GPU; it is not a
> separate install option for users.

The Python-level function tables live in `torchlight.backends.cpu`:

```python
from torchlight.backends.cpu import ARRAY_MAP, ARRAY_ZIP, ARRAY_REDUCE

ARRAY_MAP["sigmoid"]      # vectorised sigmoid on numpy arrays
ARRAY_ZIP["add"]          # np.add
ARRAY_REDUCE["max"]       # (np.maximum, -np.inf)
```

## Next

Learn about [op tracing and saving models](./06-jit-and-persistence.md).