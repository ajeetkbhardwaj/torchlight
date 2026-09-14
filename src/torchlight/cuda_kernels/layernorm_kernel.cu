// Torchlight fused layer-norm kernels (forward + backward).
//
// Clean-room rewrite of the LightSeq `layernorm_kernel.cu` semantics:
//
//   forward:  mean = mean(inp);  var = var(inp)
//             out  = (inp - mean) / sqrt(var + LN_EPSILON) * gamma + beta
//   backward: dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat*xhat)) / N) / std
//             dgamma = sum(xhat * dout);  dbeta = sum(dout)
//
// Fully self-contained: no CUB / cooperative_groups / shfl, so the same
// source compiles with nvcc (e.g. `layernorm_kernel.so` for the ctypes CUDA
// backend) AND runs under the LeetGPU simulator (`leetgpu run`).
//
// Matrices are [batch_size, hidden_dim] row-major float32. Each thread block
// handles one row; reductions use a shared-memory tree. The backward
// dgamma/dbeta kernel uses one block per hidden column.

#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>
#include <stdexcept>

#include "fused_kernels.h"

const float LN_EPSILON = 1e-8f;

// ---------------------------------------------------------------------------
// Forward kernel: one block per row.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int T_ = blockDim.x;
  const T *x = inp + row * hidden_size;
  T *o = ln_res + row * hidden_size;

  __shared__ T sred[1024];

  // -- sum and sum of squares --
  T l_sum = 0;
  T l_sq = 0;
  for (int c = tx; c < hidden_size; c += T_) {
    l_sum += x[c];
    l_sq += x[c] * x[c];
  }
  sred[tx] = l_sum;
  __syncthreads();
  for (int off = T_ >> 1; off > 0; off >>= 1) {
    if (tx < off) sred[tx] += sred[tx + off];
    __syncthreads();
  }
  const T mean = sred[0] / (T)hidden_size;
  __syncthreads();

  sred[tx] = l_sq;
  __syncthreads();
  for (int off = T_ >> 1; off > 0; off >>= 1) {
    if (tx < off) sred[tx] += sred[tx + off];
    __syncthreads();
  }
  const T var = sred[0] / (T)hidden_size - mean * mean;
  const T inv = 1.0f / sqrtf(var + LN_EPSILON);
  __syncthreads();

  if (vars) vars[row] = var;
  if (means) means[row] = mean;

  for (int c = tx; c < hidden_size; c += T_)
    o[c] = (x[c] - mean) * inv * scale[c] + bias[c];
}

// ---------------------------------------------------------------------------
// Backward dgamma/dbeta kernel: one block per hidden column.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad, const T *inp,
                                        const T *gamma, const T *betta,
                                        const T *vars, const T *means,
                                        int rows, int width) {
  const int col = blockIdx.x;
  const int tx = threadIdx.x;
  const int T_ = blockDim.x;

  __shared__ T sbeta[512];
  __shared__ T sgamma[512];

  T p_b = 0;
  T p_g = 0;
  for (int r = tx; r < rows; r += T_) {
    const T v = vars[r];
    const T m = means[r];
    const T inv = 1.0f / sqrtf(v + LN_EPSILON);
    const T xhat = (inp[r * width + col] - m) * inv;
    const T dy = out_grad[r * width + col];
    p_b += dy;
    p_g += dy * xhat;
  }
  sbeta[tx] = p_b;
  sgamma[tx] = p_g;
  __syncthreads();
  for (int off = T_ >> 1; off > 0; off >>= 1) {
    if (tx < off) {
      sbeta[tx] += sbeta[tx + off];
      sgamma[tx] += sgamma[tx + off];
    }
    __syncthreads();
  }
  if (tx == 0) {
    betta_grad[col] = sbeta[0];
    gamma_grad[col] = sgamma[0];
  }
}

// ---------------------------------------------------------------------------
// Backward dinp kernel: one block per row.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int T_ = blockDim.x;
  const T *x = inp + row * hidden_dim;
  const T *dy = out_grad + row * hidden_dim;
  T *dx = inp_grad + row * hidden_dim;

  const T mean = means ? means[row] : 0;
  const T inv = vars ? 1.0f / sqrtf(vars[row] + LN_EPSILON) : 1.0f;

  __shared__ T sdx[1024];
  __shared__ T sdxh[1024];

  T p1 = 0;
  T p2 = 0;
  for (int c = tx; c < hidden_dim; c += T_) {
    const T xhat = (x[c] - mean) * inv;
    const T d = dy[c] * gamma[c];
    p1 += d;
    p2 += d * xhat;
  }
  sdx[tx] = p1;
  sdxh[tx] = p2;
  __syncthreads();
  for (int off = T_ >> 1; off > 0; off >>= 1) {
    if (tx < off) {
      sdx[tx] += sdx[tx + off];
      sdxh[tx] += sdxh[tx + off];
    }
    __syncthreads();
  }
  const T sum_dx = sdx[0];
  const T sum_dxh = sdxh[0];
  __syncthreads();

  const T cnorm = 1.0f / (T)hidden_dim;
  for (int c = tx; c < hidden_dim; c += T_) {
    const T xhat = (x[c] - mean) * inv;
    const T d = dy[c] * gamma[c];
    dx[c] = (d - (sum_dx + xhat * sum_dxh) * cnorm) * inv;
  }
}

// ---------------------------------------------------------------------------
// Host launchers (extern "C", declared in fused_kernels.h).  They take *host*
// pointers, move the buffers to the device, run the kernel, sync and copy
// back -- exactly the reference behaviour, so the exported `.so` can be used
// straight from Python via ctypes.
// ---------------------------------------------------------------------------
void launch_layernorm(float *ln_res, float *vars, float *means,
                      const float *inp, const float *scale, const float *bias,
                      int batch_size, int hidden_dim, cudaStream_t stream) {
  if (hidden_dim <= 0 || hidden_dim > 4096) {
    throw std::runtime_error("launch_layernorm: hidden_dim out of range");
  }
  const int float_size = sizeof(float);
  const int input_size = batch_size * hidden_dim * float_size;
  const int scale_size = hidden_dim * float_size;
  const int bias_size = hidden_dim * float_size;
  const int output_size = input_size;
  const int mean_size = batch_size * float_size;
  const int var_size = batch_size * float_size;

  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  int nthread = 1;
  while (nthread < hidden_dim && nthread < 1024) nthread <<= 1;
  if (nthread < hidden_dim) {
    cudaFree(d_ln_res); cudaFree(d_vars); cudaFree(d_means);
    cudaFree(d_inp); cudaFree(d_scale); cudaFree(d_bias);
    throw std::runtime_error(
        "launch_layernorm: hidden_dim requires more than 1024 threads");
  }

  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);
}

void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp,
                         const float *gamma, const float *betta,
                         const float *vars, const float *means,
                         int batch_size, int hidden_dim, cudaStream_t stream_1,
                         cudaStream_t stream_2) {
  if (hidden_dim <= 0 || hidden_dim > 4096) {
    throw std::runtime_error("launch_layernorm_bw: hidden_dim out of range");
  }

  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp;
  float *d_gamma, *d_betta, *d_vars, *d_means;
  const int grad_output_size = batch_size * hidden_dim * sizeof(float);
  const int gamma_betta_size = hidden_dim * sizeof(float);
  const int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // dgamma/dbeta: one block per hidden column.
  int dg_threads = 256;
  dim3 grid_dim(hidden_dim);
  dim3 block_dim(dg_threads);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // dinp: one block per row.
  int nthread = 1;
  while (nthread < hidden_dim && nthread < 1024) nthread <<= 1;
  if (nthread < hidden_dim) {
    cudaFree(d_gamma_grad); cudaFree(d_betta_grad); cudaFree(d_inp_grad);
    cudaFree(d_out_grad); cudaFree(d_inp); cudaFree(d_gamma);
    cudaFree(d_betta); cudaFree(d_vars); cudaFree(d_means);
    throw std::runtime_error(
        "launch_layernorm_bw: hidden_dim requires more than 1024 threads");
  }
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means,
      hidden_dim);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}