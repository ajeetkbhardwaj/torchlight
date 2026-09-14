// Torchlight CUDA kernels -- tensorMap / tensorZip / tensorReduce / MatrixMultiply.
//
// Adapted from the MiniTorch homework skeleton (`_rough/src/combine.cu`) with the
// four `__global__` kernel bodies implemented.  The host wrappers keep the exact
// signatures the ctypes interface in `torchlight.backends` (and the reference
// `_rough/minitorch/cuda_kernel_ops.py`) expects, so the same file can be
// compiled to `combine.so` with nvcc on a CUDA box, or run directly by the
// LeetGPU simulator for validation.
//
// Op dispatch: every elementwise kernel takes a small integer `fn_id` and
// applies that scalar function (defined in `fn` below), mirroring the fn_map
// used by the Python backends.

#include <cuda_runtime.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>

#include "combine.h"

__device__ float fn(int fn_id, float x, float y = 0)
{
  switch (fn_id)
  {
  case ADD_FUNC:
    return x + y;
  case MUL_FUNC:
    return x * y;
  case ID_FUNC:
    return x;
  case NEG_FUNC:
    return -x;
  case LT_FUNC:
    return (x < y) ? 1.0f : 0.0f;
  case EQ_FUNC:
    return (x == y) ? 1.0f : 0.0f;
  case SIGMOID_FUNC:
    // Numerically stable sigmoid.
    if (x >= 0)
    {
      return 1.0f / (1.0f + exp(-x));
    }
    else
    {
      return exp(x) / (1.0f + exp(x));
    }
  case RELU_FUNC:
    return fmaxf(x, 0.0f);
  case RELU_BACK_FUNC:
    return (x > 0) ? y : 0.0f;
  case LOG_FUNC:
    return log(x + 1e-6f);
  case LOG_BACK_FUNC:
    return y / (x + 1e-6f);
  case EXP_FUNC:
    return exp(x);
  case INV_FUNC:
    return 1.0f / x;
  case INV_BACK_FUNC:
    return -(1.0f / (x * x)) * y;
  case IS_CLOSE_FUNC:
    return (x - y < 1e-2f) && (y - x < 1e-2f) ? 1.0f : 0.0f;
  case MAX_FUNC:
    return (x > y) ? x : y;
  case POW:
    return powf(x, y);
  case TANH:
    return tanhf(x);
  case SQRT_FUNC:
    return sqrtf(x);
  case ABS_FUNC:
    return fabsf(x);
  case SUB_FUNC:
    return x - y;
  case GT_FUNC:
    return (x > y) ? 1.0f : 0.0f;
  case SIGMOID_BACK_FUNC:
    return x * (1.0f - x) * y;
  case MIN_FUNC:
    return (x < y) ? x : y;
  case DIV_FUNC:
    return x / y;
  default:
    return x + y;
  }
}

// ---------------------------------------------------------------------------
// Dimension / index helpers (device)
// ---------------------------------------------------------------------------
__device__ int index_to_position(const int *index, const int *strides, int num_dims)
{
  /**
   * Converts a multidimensional tensor index into a single-dimensional position
   * in storage based on strides.
   */
  int position = 0;
  for (int i = 0; i < num_dims; ++i)
  {
    position += index[i] * strides[i];
  }
  return position;
}

__device__ void to_index(int ordinal, const int *shape, int *out_index, int num_dims)
{
  /**
   * Convert an ordinal to an index in the shape. Ensures that enumerating
   * position 0 ... size of a tensor produces every index exactly once.
   */
  int cur_ord = ordinal;
  for (int i = num_dims - 1; i >= 0; --i)
  {
    int sh = shape[i];
    out_index[i] = cur_ord % sh;
    cur_ord /= sh;
  }
}

__device__ void broadcast_index(const int *big_index, const int *big_shape, const int *shape, int *out_index, int num_dims_big, int num_dims)
{
  /**
   * Convert a big_index into big_shape to a smaller out_index into shape
   * following broadcasting rules.
   */
  for (int i = 0; i < num_dims; ++i)
  {
    if (shape[i] > 1)
    {
      out_index[i] = big_index[i + (num_dims_big - num_dims)];
    }
    else
    {
      out_index[i] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__global__ void mapKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    float *in_storage,
    int *in_shape,
    int *in_strides,
    int shape_size,
    int fn_id)
{
  /**
   * Map: apply the unary function to each element of the input and store it in
   * the (possibly shape-broadcast) output. One thread per output element.
   */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size)
  {
    return;
  }

  int out_index[MAX_DIMS];
  int in_index[MAX_DIMS];

  to_index(i, out_shape, out_index, shape_size);

  // The input may broadcast against the output (stride-0 dims / fewer dims).
  broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size);

  int o = index_to_position(out_index, out_strides, shape_size);
  int j = index_to_position(in_index, in_strides, shape_size);
  out[o] = fn(fn_id, in_storage[j]);
}

__global__ void zipKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    int out_shape_size,
    float *a_storage,
    int *a_shape,
    int *a_strides,
    int a_shape_size,
    float *b_storage,
    int *b_shape,
    int *b_strides,
    int b_shape_size,
    int fn_id)
{
  /**
   * Zip: apply the binary function pairwise to a and b (both broadcastable to
   * out) and store the result. One thread per output element.
   */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size)
  {
    return;
  }

  int out_index[MAX_DIMS];
  int a_index[MAX_DIMS];
  int b_index[MAX_DIMS];

  to_index(i, out_shape, out_index, out_shape_size);

  int o = index_to_position(out_index, out_strides, out_shape_size);

  broadcast_index(out_index, out_shape, a_shape, a_index, out_shape_size, a_shape_size);
  int j = index_to_position(a_index, a_strides, a_shape_size);

  broadcast_index(out_index, out_shape, b_shape, b_index, out_shape_size, b_shape_size);
  int k = index_to_position(b_index, b_strides, b_shape_size);

  out[o] = fn(fn_id, a_storage[j], b_storage[k]);
}

__global__ void reduceKernel(
    float *out,
    int *out_shape,
    int *out_strides,
    int out_size,
    float *a_storage,
    int *a_shape,
    int *a_strides,
    int reduce_dim,
    float reduce_value,
    int shape_size,
    int fn_id)
{
  /**
   * Reduce: reduce `a` along `reduce_dim` (output keeps that dim with size 1).
   * One thread per output element; each thread folds the reduced dimension.
   */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size)
  {
    return;
  }

  int out_index[MAX_DIMS];

  to_index(i, out_shape, out_index, shape_size);
  int o = index_to_position(out_index, out_strides, shape_size);

  int reduce_size = a_shape[reduce_dim];
  float acc = reduce_value;
  for (int s = 0; s < reduce_size; ++s)
  {
    out_index[reduce_dim] = s;
    int j = index_to_position(out_index, a_strides, shape_size);
    acc = fn(fn_id, acc, a_storage[j]);
  }
  out[o] = acc;
}

__global__ void MatrixMultiplyKernel(
    float *out,
    const int *out_shape,
    const int *out_strides,
    float *a_storage,
    const int *a_shape,
    const int *a_strides,
    float *b_storage,
    const int *b_shape,
    const int *b_strides)
{
  /**
   * Batched tiled matrix multiply. Inputs are 3D [batch, m, n] x [batch, n, p]
   * -> output [batch, m, p]. Each block (blockIdx.z = batch) computes one
   * TILE x TILE tile in shared memory.
   */
  __shared__ float a_shared[TILE][TILE];
  __shared__ float b_shared[TILE][TILE];

  int batch = blockIdx.z;
  // Support broadcast along the batch dim (a_shape[0] == 1 share a matrix).
  int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
  int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;

  int m = a_shape[1];
  int n = a_shape[2];
  int p = b_shape[2];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE + ty;  // m
  int col = blockIdx.x * TILE + tx;  // p

  int a_base = batch * a_batch_stride;
  int b_base = batch * b_batch_stride;

  float total = 0.0f;
  int n_tiles = (n + TILE - 1) / TILE;

  for (int t = 0; t < n_tiles; ++t)
  {
    // Load the a-tile [m, n] and b-tile [n, p] into shared memory.
    int a_row = blockIdx.y * TILE + ty;
    int a_col = t * TILE + tx;
    if (a_row < m && a_col < n)
    {
      a_shared[ty][tx] = a_storage[a_base + a_row * a_strides[1] + a_col * a_strides[2]];
    }
    else
    {
      a_shared[ty][tx] = 0.0f;
    }

    int b_row = t * TILE + ty;
    int b_col = blockIdx.x * TILE + tx;
    if (b_row < n && b_col < p)
    {
      b_shared[ty][tx] = b_storage[b_base + b_row * b_strides[1] + b_col * b_strides[2]];
    }
    else
    {
      b_shared[ty][tx] = 0.0f;
    }
    __syncthreads();

    for (int k2 = 0; k2 < TILE; ++k2)
    {
      total += a_shared[ty][k2] * b_shared[k2][tx];
    }
    __syncthreads();
  }

  if (row < m && col < p)
  {
    int o = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2];
    out[o] = total;
  }
}

// ---------------------------------------------------------------------------
// Host wrappers (ctypes-compatible with the Python interface)
// ---------------------------------------------------------------------------
extern "C"
{
  static void check_cuda_error(const char *what)
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
      fprintf(stderr, "%s Error: %s\n", what, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  void MatrixMultiply(
      float *out,
      int *out_shape,
      int *out_strides,
      float *a_storage,
      int *a_shape,
      int *a_strides,
      float *b_storage,
      int *b_shape,
      int *b_strides,
      int batch, int m, int p)
  {
    int n = a_shape[2];

    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc(&d_a, batch * m * n * sizeof(float));
    cudaMalloc(&d_b, batch * n * p * sizeof(float));
    cudaMalloc(&d_out, batch * m * p * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, 3 * sizeof(int));
    cudaMalloc(&d_out_strides, 3 * sizeof(int));
    cudaMalloc(&d_a_shape, 3 * sizeof(int));
    cudaMalloc(&d_a_strides, 3 * sizeof(int));
    cudaMalloc(&d_b_shape, 3 * sizeof(int));
    cudaMalloc(&d_b_strides, 3 * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, batch * m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, batch * n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    dim3 blockDims(threadsPerBlock, threadsPerBlock, 1);
    dim3 gridDims((m + threadsPerBlock - 1) / threadsPerBlock, (p + threadsPerBlock - 1) / threadsPerBlock, batch);
    MatrixMultiplyKernel<<<gridDims, blockDims>>>(
        d_out, d_out_shape, d_out_strides, d_a, d_a_shape, d_a_strides, d_b, d_b_shape, d_b_strides);

    // Copy back to the host
    cudaMemcpy(out, d_out, batch * m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check_cuda_error("Matmul");

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
  }

  void tensorMap(
      float *out,
      int *out_shape,
      int *out_strides,
      int out_size,
      float *in_storage,
      int *in_shape,
      int *in_strides,
      int in_size,
      int shape_size,
      int fn_id)
  {
    float *d_out, *d_in;
    // Allocate device memory
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_in, in_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_in_shape, *d_in_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_in_shape, shape_size * sizeof(int));
    cudaMalloc(&d_in_strides, shape_size * sizeof(int));

    // Copy data from CPU(host) to GPU(device)
    cudaMemcpy(d_in, in_storage, in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_shape, in_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_strides, in_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    mapKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size,
        d_in, d_in_shape, d_in_strides,
        shape_size, fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check_cuda_error("Map");

    // Free memory on device
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_shape);
    cudaFree(d_in_strides);
  }

  void tensorZip(
      float *out,
      int *out_shape,
      int *out_strides,
      int out_size,
      int out_shape_size,
      float *a_storage,
      int *a_shape,
      int *a_strides,
      int a_size,
      int a_shape_size,
      float *b_storage,
      int *b_shape,
      int *b_strides,
      int b_size,
      int b_shape_size,
      int fn_id)
  {
    // Allocate device memory
    float *d_out, *d_a, *d_b;
    cudaMalloc(&d_a, a_size * sizeof(float));
    cudaMalloc(&d_b, b_size * sizeof(float));
    cudaMalloc(&d_out, out_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides, *d_b_shape, *d_b_strides;
    cudaMalloc(&d_out_shape, out_shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, out_shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, a_shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, a_shape_size * sizeof(int));
    cudaMalloc(&d_b_shape, b_shape_size * sizeof(int));
    cudaMalloc(&d_b_strides, b_shape_size * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_storage, b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, out_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, a_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shape, b_shape, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_strides, b_strides, b_shape_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    zipKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size, out_shape_size,
        d_a, d_a_shape, d_a_strides, a_shape_size,
        d_b, d_b_shape, d_b_strides, b_shape_size,
        fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check_cuda_error("Zip");

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
    cudaFree(d_b_shape);
    cudaFree(d_b_strides);
  }

  void tensorReduce(
      float *out,
      int *out_shape,
      int *out_strides,
      int out_size,
      float *a_storage,
      int *a_shape,
      int *a_strides,
      int reduce_dim,
      float reduce_value,
      int shape_size,
      int fn_id)
  {
    // Allocate device memory
    int a_size = out_size * a_shape[reduce_dim];
    float *d_out, *d_a;
    cudaMalloc(&d_out, out_size * sizeof(float));
    cudaMalloc(&d_a, a_size * sizeof(float));

    int *d_out_shape, *d_out_strides, *d_a_shape, *d_a_strides;
    cudaMalloc(&d_out_shape, shape_size * sizeof(int));
    cudaMalloc(&d_out_strides, shape_size * sizeof(int));
    cudaMalloc(&d_a_shape, shape_size * sizeof(int));
    cudaMalloc(&d_a_strides, shape_size * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_a, a_storage, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_shape, out_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_strides, out_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_shape, a_shape, shape_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_strides, a_strides, shape_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (out_size + threadsPerBlock - 1) / threadsPerBlock;
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_out, d_out_shape, d_out_strides, out_size,
        d_a, d_a_shape, d_a_strides,
        reduce_dim, reduce_value, shape_size, fn_id);

    // Copy back to the host
    cudaMemcpy(out, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check_cuda_error("Reduce");

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_out_shape);
    cudaFree(d_out_strides);
    cudaFree(d_a_shape);
    cudaFree(d_a_strides);
  }
}