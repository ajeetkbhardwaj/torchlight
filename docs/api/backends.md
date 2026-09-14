# API Reference: backends

`torchlight.backends` implements the hardware layer: every tensor op reduces
to `map`, `zip`, `reduce`, and `matmul`.

```python
from torchlight.backends import CPUBackend, get_backend
```

## get_backend

```python
def get_backend(device=None)
```

Resolve a device hint into a backend instance. Accepts `None` (default
device from `TORCHLIGHT_DEVICE`, else CPU), a `Device`, a string
(`"cpu"`, `"cuda"`), or an already-instantiated backend (it is passed
through).

```python
get_backend("cpu")     # CPUBackend
get_backend("cuda")    # CudaBackend
get_backend(None)      # honours TORCHLIGHT_DEVICE
```

## CPUBackend

```python
class CPUBackend()
```

Vectorised numpy on the CPU. Every primitive operates on `TensorData`
arrays via strided views — no element-level Python loops.

| Primitive | Description |
|-----------|-------------|
| `map(fn)(a, out=None)` | unary elementwise |
| `zip(fn)(a, b)` | binary elementwise (broadcasting) |
| `reduce(ufunc, start)(a, dim)` | reduce one dimension |
| `matmul(a, b)` | (batched) matrix multiply |
| `allclose(a, b)` | numeric closeness |

The bound op names used by autograd are built in `__init__`, e.g.
`backend.sigmoid_map`, `backend.add`, `backend.relu_back`,
`backend.add_reduce`.

The vectorised kernel registry is public:

```python
from torchlight.backends.cpu import ARRAY_MAP, ARRAY_ZIP, ARRAY_REDUCE
```

| Module constant | Contents |
|-----------------|----------|
| `ARRAY_MAP` | `dict[str, fn]` — name → vectorised unary fn |
| `ARRAY_ZIP` | `dict[str, fn]` — name → vectorised binary fn |
| `ARRAY_REDUCE` | `dict[str, (ufunc, start)]` — reduction kernels |

## NumbaBackend

```python
class NumbaBackend()
```

The CPU twin of `CudaBackend`: the same kernel-style ops JIT-compiled via
numba on the CPU. Used by the test suite to develop and validate the GPU
kernels without an NVIDIA GPU — **not** a user-facing backend selection
(CPU is covered by numpy). Requires `numba`; ops without a compiled kernel
fall back to the numpy path. Also used as the fallback when numba-cuda is
unavailable.

## CudaBackend

```python
class CudaBackend()
```

numba-cuda kernels on an NVIDIA GPU. **Requires**: NVIDIA GPU + driver +
`pip install "torchlight[cuda]"` (pulls in numba-cuda). Kernels transfer to
the GPU and back per op. Raises a clear install hint if numba/cuda is not
available on the machine.

When the compiled CUDA C kernel shared objects
(`softmax_kernel.so` / `layernorm_kernel.so`) are present in
`torchlight/cuda_kernels/` (or pointed at via `TORCHLIGHT_CUDA_KERNEL_DIR`),
the fused ops are routed through `CudaCKernelProvider` — a subclass of
`CudaProvider` that loads the `.so` launchers via ctypes and delegates only
`attn_softmax_fw/bw` and `layernorm_fw/bw` to them; everything else still
runs the numba-cuda kernels.

!!! warning
    Verified against the `leetgpu` simulator; not yet tested on real NVIDIA
    hardware. Mirrors the numba kernels structurally.

## Singletons

```python
from torchlight.backends.cuda import numba_backend, cuda_backend
```

`numba_backend()` / `cuda_backend()` return cached singletons (so repeated
`get_backend("cuda")` calls share state).

## Device

Each backend exposes `.device` (`Device("cpu")` for CPU/numba;
`Device("cuda")` for CUDA) and `Tensor.device` reports the backend's
device. See [core device API](./core.md#device).