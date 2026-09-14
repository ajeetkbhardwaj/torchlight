// Torchlight fused attention-softmax kernels (forward + backward).
//
// Clean-room rewrite of the LightSeq `softmax_kernel.cu` semantics:
//
//   forward:  out = exp(x + m - max(x + m)) / (sum + 1e-8)   (last axis)
//   backward: d   = s * (g - sum(g * s))                     (last axis)
//
// Unlike the LightSeq skeleton this is fully self-contained: no CUB and no
// cooperative_groups, so the same source compiles with nvcc (producing e.g.
// `softmax_kernel.so` for the ctypes CUDA backend) AND runs under the
// LeetGPU simulator for validation (`leetgpu run`).
//
// Memory layout / indexing (matches LightSeq):
//   inp:  [batch_size, nhead, from_len, to_len]  (row-major, float32)
//   mask: [batch_size, to_len]                   (values ~ -1e8 for masked)
// One thread block handles one (batch, head) pair and loops over the token
// rows; columns are reduced with a shared-memory tree inside the block.

#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>
#include <stdexcept>

#include "fused_kernels.h"

const float EPSILON = 1e-8f;
const float F_INF_NEG = -100000000.f;

__forceinline__ __device__ int ln_3dim(int id1, int id2, int id3, int dim2,
                                       int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

// ---------------------------------------------------------------------------
// Forward kernel: one block per (batch_id, head_id); blockDim is a power of
// two covering `to_len` (<= 1024).  Each block iterates the `from_len` token
// rows so `gridDim.x` stays 1 (the launcher keeps the reference grid shape).
// ---------------------------------------------------------------------------
__global__ void ker_attn_softmax(float *inp, const float *attn_mask,
                                 int from_len, int to_len, bool mask_future) {
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int tx = threadIdx.x;
  const int T = blockDim.x;

  __shared__ float sval[1024];
  __shared__ float sred[1024];

  float *base =
      inp + ln_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  const float *mask_row = attn_mask ? attn_mask + batch_id * to_len : nullptr;

  for (int token_id = blockIdx.x; token_id < from_len;
       token_id += gridDim.x) {
    float *row = base + token_id * to_len;

    // -- step 1: row max (softmax input = row + mask, future tokens masked) --
    float lmax = F_INF_NEG;
    for (int c = tx; c < to_len; c += T) {
      float v = row[c];
      if (mask_row) v += mask_row[c];
      if (mask_future && c > token_id) v = F_INF_NEG;
      sval[c] = v;
      lmax = fmaxf(lmax, v);
    }
    sred[tx] = lmax;
    __syncthreads();
    for (int off = T >> 1; off > 0; off >>= 1) {
      if (tx < off) sred[tx] = fmaxf(sred[tx], sred[tx + off]);
      __syncthreads();
    }
    const float mval = sred[0];
    __syncthreads();

    // -- step 2: sum of exp(x - max) --
    float lsum = 0.0f;
    for (int c = tx; c < to_len; c += T) {
      const float e = expf(sval[c] - mval);
      sval[c] = e;
      lsum += e;
    }
    sred[tx] = lsum;
    __syncthreads();
    for (int off = T >> 1; off > 0; off >>= 1) {
      if (tx < off) sred[tx] += sred[tx + off];
      __syncthreads();
    }
    const float inv = 1.0f / (sred[0] + EPSILON);
    __syncthreads();

    // -- step 3: write normalised row --
    for (int c = tx; c < to_len; c += T) row[c] = sval[c] * inv;
    __syncthreads();  // protect `sval` reuse across token rows
  }
}

// ---------------------------------------------------------------------------
// Backward kernel: one block per row; reduces sum(g * s) over the last axis.
// out_grad[rows, to_len] is updated in place:
//   out_grad[r, c] = soft[r, c] * (out_grad[r, c] - sum_c(soft * out_grad))
// ---------------------------------------------------------------------------
__global__ void ker_attn_softmax_bw(float *out_grad, const float *soft,
                                    int softmax_length) {
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int T = blockDim.x;
  float *g = out_grad + row * softmax_length;
  const float *s = soft + row * softmax_length;

  __shared__ float sred[1024];

  float lsum = 0.0f;
  for (int c = tx; c < softmax_length; c += T) lsum += g[c] * s[c];
  sred[tx] = lsum;
  __syncthreads();
  for (int off = T >> 1; off > 0; off >>= 1) {
    if (tx < off) sred[tx] += sred[tx + off];
    __syncthreads();
  }
  const float total = sred[0];
  __syncthreads();

  for (int c = tx; c < softmax_length; c += T) g[c] = s[c] * (g[c] - total);
}

// ---------------------------------------------------------------------------
// Host launchers (extern "C", declared in fused_kernels.h).  They take *host*
// pointers, move the buffers to the device, run the kernel, sync and copy
// back -- exactly the reference behaviour, so the exported `.so` can be used
// straight from Python via ctypes.
// ---------------------------------------------------------------------------
void launch_attn_softmax(float *inp, const float *attn_mask, int batch_size,
                         int nhead, int from_len, int to_len, bool mask_future,
                         cudaStream_t stream) {
  if (to_len <= 0 || to_len > 1024) {
    throw std::runtime_error(
        "launch_attn_softmax: to_len must be within (0, 1024]");
  }
  const int inp_size = batch_size * nhead * from_len * to_len;
  float *d_inp = nullptr;
  cudaMalloc(&d_inp, inp_size * sizeof(float));
  cudaMemcpy(d_inp, inp, inp_size * sizeof(float), cudaMemcpyHostToDevice);

  float *d_mask = nullptr;
  if (attn_mask) {
    cudaMalloc(&d_mask, batch_size * to_len * sizeof(float));
    cudaMemcpy(d_mask, attn_mask, batch_size * to_len * sizeof(float),
               cudaMemcpyHostToDevice);
  }

  dim3 grid(1, batch_size, nhead);
  int threads = 1;
  while (threads < to_len && threads < 1024) threads <<= 1;
  if (threads < to_len) {
    cudaFree(d_inp);
    if (d_mask) cudaFree(d_mask);
    throw std::runtime_error(
        "launch_attn_softmax: to_len requires more than 1024 threads");
  }

  ker_attn_softmax<<<grid, threads, 0, stream>>>(d_inp, d_mask, from_len,
                                                 to_len, mask_future);

  cudaMemcpy(inp, d_inp, inp_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_attn_softmax error: %s\n",
            cudaGetErrorString(err));
  }

  cudaFree(d_inp);
  if (d_mask) cudaFree(d_mask);
}

void launch_attn_softmax_bw(float *out_grad, const float *soft_inp, int rows,
                            int softmax_len, cudaStream_t stream) {
  if (softmax_len <= 0 || softmax_len > 1024) {
    throw std::runtime_error(
        "launch_attn_softmax_bw: softmax_len must be within (0, 1024]");
  }
  const int size = rows * softmax_len;
  float *d_grad = nullptr;
  float *d_soft = nullptr;
  cudaMalloc(&d_grad, size * sizeof(float));
  cudaMalloc(&d_soft, size * sizeof(float));
  cudaMemcpy(d_grad, out_grad, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_soft, soft_inp, size * sizeof(float), cudaMemcpyHostToDevice);

  int threads = 1;
  while (threads < softmax_len && threads < 1024) threads <<= 1;
  if (threads < softmax_len) {
    cudaFree(d_grad);
    cudaFree(d_soft);
    throw std::runtime_error(
        "launch_attn_softmax_bw: softmax_len requires more than 1024 threads");
  }

  ker_attn_softmax_bw<<<rows, threads, 0, stream>>>(d_grad, d_soft,
                                                    softmax_len);

  cudaMemcpy(out_grad, d_grad, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_attn_softmax_bw error: %s\n",
            cudaGetErrorString(err));
  }

  cudaFree(d_grad);
  cudaFree(d_soft);
}