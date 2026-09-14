# Example: CUDA Kernels (Reference)

`examples/cuda/` holds the **reference CUDA tests** for torchlight's GPU
backend. They are not Python examples -- they exercise the same kernels that
`torchlight.cuda.Kernels.SUPPORTED_OPS` dispatches to at runtime, so the
kernel behaviour can be validated on a host with no GPU attached.

## What is in here

* `tensor_cuda_selftest.cu` -- exercises `tensorMap` / `tensorZip` /
  `tensorReduce` / `MatrixMultiply` (the four backend primitives) against
  reference host computations, using the same `fn_id` dispatch constants as
  `src/torchlight/backends/cuda.py`.
* `fused_kernels_test.cu` -- the fused softmax / LayerNorm kernels
  (mirrored as numba-JIT kernels alongside `softmax_kernel.cu` /
  `layernorm_kernel.cu`).

The primitives are the ones [Tutorial 5](../tutorials/05-backends.md)
describes: **map / zip / reduce / matmul**. The `.cu` sources under
`src/torchlight/cuda_kernels/` are *not* shipped in the pip wheel -- nothing at
runtime needs them, because the numba-mirrored fused ops and per-op numpy are
always available.

## Running the self-tests

With [LeetGPU](https://github.com/realminchoi/LeetGPU):

```
leetgpu run src/torchlight/cuda_kernels/combine.h \
              src/torchlight/cuda_kernels/combine.cu \
              examples/cuda/tensor_cuda_selftest.cu
```

On a real CUDA host you can instead compile the kernels to a `.so` and let the
`cuda_backend` load them via ctypes (the code path described in
[Tutorial 5](../tutorials/05-backends.md) and `backends/cuda.py`).

## What this example teaches

- the four-primitive backend contract is small enough to implement in CUDA-C
  by hand,
- `fn_id` dispatch keeps the Python/numba/CUDA op tables in lock-step,
- GPU code can be validated CPU-side with a kernel self-test before hardware
  exists.