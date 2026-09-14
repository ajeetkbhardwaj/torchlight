// Torchlight fused CUDA kernel launchers (attention-softmax / layer-norm).
//
// Pure declarations, shared by:
//   * softmax_kernel.cu   / layernorm_kernel.cu (the definitions),
//   * examples/cuda/fused_kernels_test.cu       (the LeetGPU harness).
//
// LeetGPU run (mirrors the combine.cu workflow):
//
//   leetgpu run examples/cuda/fused_kernels_test.cu \
//              src/torchlight/cuda_kernels/fused_kernels.h \
//              src/torchlight/cuda_kernels/softmax_kernel.cu \
//              src/torchlight/cuda_kernels/layernorm_kernel.cu

#pragma once

#include <cuda_runtime.h>

extern "C" {

// Forward (in place): inp[b, h, t, :] <- softmax(inp + mask[b, :]), mask_future
// masks columns c > t.  inp is [batch, nhead, from_len, to_len]; mask is
// [batch, to_len] (values ~ -1e8 for masked positions).
void launch_attn_softmax(float *inp, const float *attn_mask, int batch_size,
                         int nhead, int from_len, int to_len, bool mask_future,
                         cudaStream_t stream);

// Backward (in place): out_grad[r, :] <- s * (g - sum(s * g)); soft is
// [rows, softmax_len], already the softmax output of the forward.
void launch_attn_softmax_bw(float *out_grad, const float *soft_inp, int rows,
                            int softmax_len, cudaStream_t stream);

// Forward: ln_res[b, :] <- (inp[b, :] - mean) / sqrt(var + eps) * scale + bias.
// vars/means are side outputs ([batch]); may be nullptr to skip.
void launch_layernorm(float *ln_res, float *vars, float *means,
                      const float *inp, const float *scale, const float *bias,
                      int batch_size, int hidden_dim, cudaStream_t stream);

// Backward: dgamma = sum(xhat * dout), dbeta = sum(dout),
// dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden) / std.
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp,
                         const float *gamma, const float *betta,
                         const float *vars, const float *means, int batch_size,
                         int hidden_dim, cudaStream_t stream_1,
                         cudaStream_t stream_2);
}