// LeetGPU example harness for the fused attention-softmax and layer-norm
// kernels (softmax_kernel.cu / layernorm_kernel.cu).
//
// Run with:
//   leetgpu run examples/cuda/fused_kernels_test.cu \
//              src/torchlight/cuda_kernels/fused_kernels.h \
//              src/torchlight/cuda_kernels/softmax_kernel.cu \
//              src/torchlight/cuda_kernels/layernorm_kernel.cu
//
// It calls the exported extern "C" launchers (declared in fused_kernels.h) on
// small random tensors and compares the results against the host-side
// reference math, printing PASS / FAIL per check.  The process exits non-zero
// if any check fails.

#include <math.h>
#include <cstdio>
#include <cstdlib>

#include "fused_kernels.h"

static const float ATTN_EPS = 1e-8f;
static const float LN_EPS = 1e-8f;

// ---------------------------------------------------------------------------
// small deterministic PRNG
// ---------------------------------------------------------------------------
static unsigned int g_seed = 12345u;
static float frand() {
  g_seed = g_seed * 1103515245u + 12345u;
  return ((float)((g_seed >> 9) & 0xffff) / 65535.0f) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// host references
// ---------------------------------------------------------------------------
static void ref_attn_softmax_fw(float *out, const float *inp,
                                const float *mask, int B, int H, int Tt,
                                int S, bool mask_future) {
  const int R = B * H * Tt;
  for (int r = 0; r < R; r++) {
    const int b = (r / (H * Tt)) % B;
    const int t = r % Tt;
    const float *row = inp + r * S;
    float mx = -1e30f;
    for (int c = 0; c < S; c++) {
      float v = row[c] + (mask ? mask[b * S + c] : 0.0f);
      if (mask_future && c > t) v = -1e8f;
      if (v > mx) mx = v;
    }
    float sm = 0.0f;
    for (int c = 0; c < S; c++) {
      float v = row[c] + (mask ? mask[b * S + c] : 0.0f);
      if (mask_future && c > t) v = -1e8f;
      out[r * S + c] = expf(v - mx);
      sm += out[r * S + c];
    }
    const float inv = 1.0f / (sm + ATTN_EPS);
    for (int c = 0; c < S; c++) out[r * S + c] *= inv;
  }
}

static void ref_attn_softmax_bw(float *out, const float *soft, const float *g,
                                int rows, int S) {
  for (int r = 0; r < rows; r++) {
    const float *s = soft + r * S;
    const float *gy = g + r * S;
    float tot = 0.0f;
    for (int c = 0; c < S; c++) tot += gy[c] * s[c];
    for (int c = 0; c < S; c++) out[r * S + c] = s[c] * (gy[c] - tot);
  }
}

static void ref_layernorm_fw(float *out, float *vars, float *means,
                             const float *inp, const float *scale,
                             const float *bias, int rows, int D) {
  for (int r = 0; r < rows; r++) {
    const float *x = inp + r * D;
    float mean = 0.0f, sq = 0.0f;
    for (int c = 0; c < D; c++) { mean += x[c]; sq += x[c] * x[c]; }
    mean /= (float)D;
    const float var = sq / (float)D - mean * mean;
    const float inv = 1.0f / sqrtf(var + LN_EPS);
    if (vars) vars[r] = var;
    if (means) means[r] = mean;
    for (int c = 0; c < D; c++) out[r * D + c] = (x[c] - mean) * inv * scale[c] + bias[c];
  }
}

static void ref_layernorm_bw(float *dinp, float *dgamma, float *dbeta,
                             const float *dy, const float *x, const float *gamma,
                             const float *var, const float *mean, int rows, int D) {
  for (int c = 0; c < D; c++) { dgamma[c] = 0.0f; dbeta[c] = 0.0f; }
  for (int r = 0; r < rows; r++) {
    const float inv = 1.0f / sqrtf(var[r] + LN_EPS);
    float sd = 0.0f, sdxh = 0.0f;
    for (int c = 0; c < D; c++) {
      const float xhat = (x[r * D + c] - mean[r]) * inv;
      const float d = dy[r * D + c] * gamma[c];
      sd += d;
      sdxh += d * xhat;
      dgamma[c] += xhat * dy[r * D + c];
      dbeta[c] += dy[r * D + c];
    }
    const float cnorm = 1.0f / (float)D;
    for (int c = 0; c < D; c++) {
      const float xhat = (x[r * D + c] - mean[r]) * inv;
      const float d = dy[r * D + c] * gamma[c];
      dinp[r * D + c] = (d - (sd + xhat * sdxh) * cnorm) * inv;
    }
  }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static float max_abs_diff(const float *a, const float *b, int n) {
  float mx = 0.0f;
  for (int i = 0; i < n; i++) {
    const float d = fabsf(a[i] - b[i]);
    if (d > mx) mx = d;
  }
  return mx;
}

static int g_failures = 0;

static void check(const char *name, float got, float tol) {
  const bool ok = got <= tol;
  printf("  [%s] %s (max abs diff %.3e)\n", ok ? "PASS" : "FAIL", name, got);
  if (!ok) g_failures++;
}

// ---------------------------------------------------------------------------
// test cases
// ---------------------------------------------------------------------------
static void test_attn_softmax_fw() {
  printf("test_attn_softmax_fw\n");
  int shapes[][4] = {{2, 1, 4, 8},   {1, 2, 3, 16}, {1, 1, 5, 5},
                     {2, 1, 6, 64},  {1, 3, 2, 100}};
  int masks[][2] = {{1, 0}, {0, 1}, {0, 0}, {1, 1}, {0, 1}};
  const int ncases = 5;
  for (int t = 0; t < ncases; t++) {
    const int B = shapes[t][0], H = shapes[t][1], Tt = shapes[t][2], S = shapes[t][3];
    const bool mf = masks[t][0] == 1;
    const bool usemask = masks[t][1] == 1;
    const int N = B * H * Tt * S;
    float *inp = (float *)malloc(N * sizeof(float));
    float *out = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));
    float *mask = usemask ? (float *)malloc(B * S * sizeof(float)) : nullptr;
    for (int i = 0; i < N; i++) inp[i] = frand() * 4.0f;
    if (usemask)
      for (int i = 0; i < B * S; i++) mask[i] = (frand() > 0.0f) ? 0.0f : -1e8f;
    memcpy(out, inp, N * sizeof(float));
    launch_attn_softmax(out, mask, B, H, Tt, S, mf, 0);
    ref_attn_softmax_fw(ref, inp, mask, B, H, Tt, S, mf);
    char nm[64];
    snprintf(nm, 64, "fw B=%d H=%d T=%d S=%d mf=%d", B, H, Tt, S, (int)mf);
    check(nm, max_abs_diff(out, ref, N), 2e-3f);
    // row sum ~ 1 sanity
    float worst = 0.0f;
    for (int r = 0; r < B * H * Tt; r++) {
      float sm = 0.0f;
      for (int c = 0; c < S; c++) sm += out[r * S + c];
      const float d = fabsf(sm - 1.0f);
      if (d > worst) worst = d;
    }
    check("  rowsum", worst, 2e-3f);
    free(inp); free(out); free(ref); free(mask);
  }
}

static void test_attn_softmax_bw() {
  printf("test_attn_softmax_bw\n");
  int shapes[][2] = {{8, 16}, {5, 5}, {64, 64}, {3, 100}, {1, 1024}};
  for (int t = 0; t < 5; t++) {
    const int rows = shapes[t][0], S = shapes[t][1];
    const int N = rows * S;
    float *soft = (float *)malloc(N * sizeof(float));
    float *g = (float *)malloc(N * sizeof(float));
    float *got = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));
    for (int r = 0; r < rows; r++) {
      float sm = 0.0f;
      for (int c = 0; c < S; c++) { soft[r * S + c] = fabsf(frand()) + 1e-3f; sm += soft[r * S + c]; }
      for (int c = 0; c < S; c++) soft[r * S + c] /= sm;
    }
    for (int i = 0; i < N; i++) g[i] = frand();
    memcpy(got, g, N * sizeof(float));
    launch_attn_softmax_bw(got, soft, rows, S, 0);
    ref_attn_softmax_bw(ref, soft, g, rows, S);
    char nm[64];
    snprintf(nm, 64, "bw rows=%d S=%d", rows, S);
    check(nm, max_abs_diff(got, ref, N), 2e-3f);
    free(soft); free(g); free(got); free(ref);
  }
}

static void test_layernorm() {
  printf("test_layernorm\n");
  int shapes[][2] = {{4, 128}, {16, 17}, {2, 5}, {8, 1024}, {3, 3}};
  for (int t = 0; t < 5; t++) {
    const int rows = shapes[t][0], D = shapes[t][1];
    const int N = rows * D;
    float *inp = (float *)malloc(N * sizeof(float));
    float *gamma = (float *)malloc(D * sizeof(float));
    float *beta = (float *)malloc(D * sizeof(float));
    float *out = (float *)malloc(N * sizeof(float));
    float *ref = (float *)malloc(N * sizeof(float));
    float *vars = (float *)malloc(rows * sizeof(float));
    float *means = (float *)malloc(rows * sizeof(float));
    for (int i = 0; i < N; i++) inp[i] = frand() * 2.0f + 1.0f;
    for (int i = 0; i < D; i++) { gamma[i] = frand() * 0.5f + 0.75f; beta[i] = frand(); }
    launch_layernorm(out, vars, means, inp, gamma, beta, rows, D, 0);
    ref_layernorm_fw(ref, nullptr, nullptr, inp, gamma, beta, rows, D);
    char nm[64];
    snprintf(nm, 64, "fw rows=%d D=%d", rows, D);
    check(nm, max_abs_diff(out, ref, N), 2e-3f);

    // backward
    float *dy = (float *)malloc(N * sizeof(float));
    float *dinp = (float *)malloc(N * sizeof(float));
    float *rdinp = (float *)malloc(N * sizeof(float));
    float *dgamma = (float *)malloc(D * sizeof(float));
    float *rdgamma = (float *)malloc(D * sizeof(float));
    float *dbeta = (float *)malloc(D * sizeof(float));
    float *rdbeta = (float *)malloc(D * sizeof(float));
    for (int i = 0; i < N; i++) dy[i] = frand();

    // recompute ref vars/means (same LN_EPS path)
    ref_layernorm_fw(ref, vars, means, inp, gamma, beta, rows, D);
    ref_layernorm_bw(rdinp, rdgamma, rdbeta, dy, inp, gamma, vars, means, rows, D);

    memset(dinp, 0, N * sizeof(float));
    memset(dgamma, 0, D * sizeof(float));
    memset(dbeta, 0, D * sizeof(float));
    launch_layernorm_bw(dgamma, dbeta, dinp, dy, inp, gamma, beta, vars, means,
                        rows, D, 0, 0);
    snprintf(nm, 64, "bw dinp rows=%d D=%d", rows, D);
    check(nm, max_abs_diff(dinp, rdinp, N), 3e-3f);
    snprintf(nm, 64, "bw dgamma rows=%d D=%d", rows, D);
    check(nm, max_abs_diff(dgamma, rdgamma, D), 3e-3f);
    snprintf(nm, 64, "bw dbeta rows=%d D=%d", rows, D);
    check(nm, max_abs_diff(dbeta, rdbeta, D), 3e-3f);

    free(inp); free(gamma); free(beta); free(out); free(ref);
    free(vars); free(means); free(dy); free(dinp); free(rdinp);
    free(dgamma); free(rdgamma); free(dbeta); free(rdbeta);
  }
}

int main() {
  test_attn_softmax_fw();
  test_attn_softmax_bw();
  test_layernorm();
  if (g_failures == 0) {
    printf("ALL FUSED KERNEL CHECKS PASSED\n");
    return 0;
  }
  printf("%d CHECK(S) FAILED\n", g_failures);
  return 1;
}