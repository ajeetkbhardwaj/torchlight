// LeetGPU self-test for torchlight's CUDA kernels (combine.cu).
//
// Run with:
//   leetgpu run src/torchlight/cuda_kernels/combine.h \
//            src/torchlight/cuda_kernels/combine.cu \
//            examples/cuda/tensor_cuda_selftest.cu
//
// Exercises tensorMap / tensorZip / tensorReduce / MatrixMultiply against
// reference computations performed on the "host" (i.e. the simulator), using
// the exact same fn_id dispatch constants the kernels use.

#include "combine.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg)                                          \
  do                                                             \
  {                                                              \
    if (!(cond))                                                 \
    {                                                            \
      printf("  FAIL %s\n", msg);                                \
      failures++;                                                \
    }                                                            \
    else                                                         \
    {                                                            \
      printf("  ok   %s\n", msg);                                \
    }                                                            \
  } while (0)

// Reference scalar ops (host implementation mirroring the kernel's `fn`).
static float host_fn(int fn_id, float x, float y)
{
  switch (fn_id)
  {
  case ADD_FUNC:
    return x + y;
  case MUL_FUNC:
    return x * y;
  case SIGMOID_FUNC:
    if (x >= 0)
      return 1.0f / (1.0f + expf(-x));
    return expf(x) / (1.0f + expf(x));
  case RELU_FUNC:
    return x > 0 ? x : 0.0f;
  case EXP_FUNC:
    return expf(x);
  case MAX_FUNC:
    return x > y ? x : y;
  case SQRT_FUNC:
    return sqrtf(x);
  case ABS_FUNC:
    return fabsf(x);
  case SUB_FUNC:
    return x - y;
  case MIN_FUNC:
    return x < y ? x : y;
  default:
    return x;
  }
}

static int close_enough(float a, float b)
{
  float tol = 1e-4f;
  return fabsf(a - b) <= tol;
}

static void run_map_test()
{
  printf("== tensorMap (sigmoid, then broadcast relu) ==\n");

  // a: shape (2, 3), contiguous
  float a[6] = {-1.0f, 0.0f, 0.5f, 3.0f, -3.0f, 4.0f};
  int a_shape[2] = {2, 3};
  int a_strides[2] = {3, 1};
  int shape_size = 2;

  float expect[6];
  for (int i = 0; i < 6; i++)
  {
    expect[i] = host_fn(SIGMOID_FUNC, a[i], 0);
  }

  float out[6];
  int out_shape[2] = {2, 3};
  int out_strides[2] = {3, 1};
  tensorMap(out, out_shape, out_strides, 6, a, a_shape, a_strides, 6, shape_size, SIGMOID_FUNC);

  int ok = 1;
  for (int i = 0; i < 6; i++)
  {
    if (!close_enough(out[i], expect[i]))
    {
      printf("    out[%d]=%f expect %f\n", i, out[i], expect[i]);
      ok = 0;
    }
  }
  CHECK(ok, "sigmoid map matches host");

  // Broadcast map: input (1, 3) -> output (2, 3), relu.
  float small[3] = {-1.0f, 2.0f, -0.5f};
  int small_shape[2] = {1, 3};
  int small_strides[2] = {0, 1};
  float out2[6];
  tensorMap(out2, out_shape, out_strides, 6, small, small_shape, small_strides, 3, shape_size, RELU_FUNC);

  int ok2 = 1;
  for (int i = 0; i < 6; i++)
  {
    float e = host_fn(RELU_FUNC, small[i % 3], 0);
    if (!close_enough(out2[i], e))
    {
      printf("    out2[%d]=%f expect %f\n", i, out2[i], e);
      ok2 = 0;
    }
  }
  CHECK(ok2, "strided broadcast map matches host");

  // sqrt map over the same input.
  float out3[6];
  tensorMap(out3, out_shape, out_strides, 6, a, a_shape, a_strides, 6, shape_size, SQRT_FUNC);
  int ok3 = 1;
  for (int i = 0; i < 6; i++)
  {
    float e = host_fn(SQRT_FUNC, fabsf(a[i]), 0);  // sqrt defined for negatives-free data
    if (a[i] < 0)
    {
      continue;  // skip NaN-producing entries
    }
    if (!close_enough(out3[i], e))
    {
      printf("    out3[%d]=%f expect %f\n", i, out3[i], e);
      ok3 = 0;
    }
  }
  CHECK(ok3, "sqrt map matches host");
}

static void run_zip_test()
{
  printf("== tensorZip (add) ==\n");

  // a: (2,3), b: (2,3) same shape -> elementwise add
  float a[6] = {1, 2, 3, 4, 5, 6};
  float b[6] = {10, 20, 30, 40, 50, 60};
  int shape2[2] = {2, 3};
  int st[2] = {3, 1};

  float out[6];
  float expect[6];
  for (int i = 0; i < 6; i++)
  {
    expect[i] = a[i] + b[i];
  }

  tensorZip(out, shape2, st, 6, 2, a, shape2, st, 6, 2, b, shape2, st, 6, 2, ADD_FUNC);

  int ok = 1;
  for (int i = 0; i < 6; i++)
  {
    if (!close_enough(out[i], expect[i]))
    {
      printf("    out[%d]=%f expect %f\n", i, out[i], expect[i]);
      ok = 0;
    }
  }
  CHECK(ok, "zip add matches host");

  // Broadcast zip: bias (3,) onto (2,3).
  float bias[3] = {1, 2, 3};
  int bias_shape[1] = {3};
  int bias_st[1] = {1};
  float out2[6];

  tensorZip(out2, shape2, st, 6, 2, a, shape2, st, 6, 2, bias, bias_shape, bias_st, 3, 1, ADD_FUNC);

  int ok2 = 1;
  for (int i = 0; i < 6; i++)
  {
    float e = a[i] + bias[i % 3];
    if (!close_enough(out2[i], e))
    {
      printf("    out2[%d]=%f expect %f\n", i, out2[i], e);
      ok2 = 0;
    }
  }
  CHECK(ok2, "broadcast zip add matches host");

  // subtraction zip (bias) for parity with sub op
  float out3[6];
  tensorZip(out3, shape2, st, 6, 2, a, shape2, st, 6, 2, bias, bias_shape, bias_st, 3, 1, SUB_FUNC);
  int ok3 = 1;
  for (int i = 0; i < 6; i++)
  {
    float e = a[i] - bias[i % 3];
    if (!close_enough(out3[i], e))
    {
      printf("    out3[%d]=%f expect %f\n", i, out3[i], e);
      ok3 = 0;
    }
  }
  CHECK(ok3, "broadcast zip sub matches host");
}

static void run_reduce_test()
{
  printf("== tensorReduce (sum along dim 1, then max along dim 0) ==\n");

  // a: (2,3) -> reduce dim1 -> (2,1)
  float a[6] = {1, 2, 3, 4, 5, 6};
  int a_shape[2] = {2, 3};
  int a_strides[2] = {3, 1};

  int out_shape[2] = {2, 1};
  int out_strides[2] = {1, 1};
  float out[2];

  tensorReduce(out, out_shape, out_strides, 2, a, a_shape, a_strides, 1, 0.0f, 2, ADD_FUNC);

  float e0 = a[0] + a[1] + a[2];
  float e1 = a[3] + a[4] + a[5];
  CHECK(close_enough(out[0], e0) && close_enough(out[1], e1), "sum reduce -> (2,1)");

  // max along dim0 -> (1,3)
  int out_shape2[2] = {1, 3};
  int out_strides2[2] = {3, 1};
  float out2[3];
  tensorReduce(out2, out_shape2, out_strides2, 3, a, a_shape, a_strides, 0, -1e30f, 2, MAX_FUNC);

  float em[3] = {a[0] > a[3] ? a[0] : a[3],
                 a[1] > a[4] ? a[1] : a[4],
                 a[2] > a[5] ? a[2] : a[5]};
  int ok2 = 1;
  for (int i = 0; i < 3; i++)
  {
    if (!close_enough(out2[i], em[i]))
    {
      printf("    out2[%d]=%f expect %f\n", i, out2[i], em[i]);
      ok2 = 0;
    }
  }
  CHECK(ok2, "max reduce -> (1,3)");

  // min reduce along dim0 -> (1,3)
  float out3[3];
  tensorReduce(out3, out_shape2, out_strides2, 3, a, a_shape, a_strides, 0, 1e30f, 2, MIN_FUNC);
  int ok3 = 1;
  for (int i = 0; i < 3; i++)
  {
    float e = a[i] < a[i + 3] ? a[i] : a[i + 3];
    if (!close_enough(out3[i], e))
    {
      printf("    out3[%d]=%f expect %f\n", i, out3[i], e);
      ok3 = 0;
    }
  }
  CHECK(ok3, "min reduce -> (1,3)");
}

static void run_matmul_test()
{
  printf("== MatrixMultiply ==\n");

  // 2D case: a (2,3) x b (3,2) -> (2,2), batched as batch=1.
  float a[6] = {1, 2, 3, 4, 5, 6};
  float b[6] = {7, 8, 9, 10, 11, 12};
  int m = 2, n = 3, p = 2;

  int a_shape[3] = {1, m, n};   // batch, m, n
  int b_shape[3] = {1, n, p};   // batch, n, p
  int a_st[3] = {6, 3, 1};
  int b_st[3] = {6, 2, 1};

  int out_shape[3] = {1, m, p};
  int out_st[3] = {4, 2, 1};
  float out[4];
  MatrixMultiply(out, out_shape, out_st, a, a_shape, a_st, b, b_shape, b_st, 1, m, p);

  float em[4] = {1 * 7 + 2 * 9 + 3 * 11, 1 * 8 + 2 * 10 + 3 * 12,
                 4 * 7 + 5 * 9 + 6 * 11, 4 * 8 + 5 * 10 + 6 * 12};
  int ok = 1;
  for (int i = 0; i < 4; i++)
  {
    if (!close_enough(out[i], em[i]))
    {
      printf("    out[%d]=%f expect %f\n", i, out[i], em[i]);
      ok = 0;
    }
  }
  CHECK(ok, "2D matmul");

  // Batched 3D: batch=2 of (2,3) x (3,2).
  float abatch[12] = {1, 2, 3, 4, 5, 6,
                      7, 8, 9, 10, 11, 12};
  float bbatch[12] = {1, 0, 0, 1, 1, 1,
                      2, 0, 0, 2, 1, 1};
  int ab_shape[3] = {2, m, n};
  int bb_shape[3] = {2, n, p};
  int ab_st[3] = {6, 3, 1};
  int bb_st[3] = {6, 2, 1};
  int out2_shape[3] = {2, m, p};
  int out2_st[3] = {4, 2, 1};
  float out2[8];

  MatrixMultiply(out2, out2_shape, out2_st, abatch, ab_shape, ab_st, bbatch, bb_shape, bb_st, 2, m, p);

  int okb = 1;
  for (int batch = 0; batch < 2 && okb; batch++)
  {
    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < p; j++)
      {
        float e = 0;
        for (int k = 0; k < n; k++)
        {
          e += abatch[batch * m * n + i * n + k] * bbatch[batch * n * p + k * p + j];
        }
        float g = out2[batch * m * p + i * p + j];
        if (!close_enough(g, e))
        {
          printf("    out2[%d,%d,%d]=%f expect %f\n", batch, i, j, g, e);
          okb = 0;
        }
      }
    }
  }
  CHECK(okb, "batched 3D matmul (batch=2)");

  // Broadcast batch: b as batch=1 shared across a's 2 batches.
  MatrixMultiply(out2, out2_shape, out2_st, abatch, ab_shape, ab_st, b, b_shape, b_st, 2, m, p);
  int okc = 1;
  for (int batch = 0; batch < 2 && okc; batch++)
  {
    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < p; j++)
      {
        float e = 0;
        for (int k = 0; k < n; k++)
        {
          e += abatch[batch * m * n + i * n + k] * b[k * p + j];
        }
        float g = out2[batch * m * p + i * p + j];
        if (!close_enough(g, e))
        {
          printf("    out2[%d,%d,%d]=%f expect %f\n", batch, i, j, g, e);
          okc = 0;
        }
      }
    }
  }
  CHECK(okc, "matrix broadcast over batch dim");
}

int main()
{
  printf("torchlight CUDA kernels self-test (LeetGPU)\n");
  run_map_test();
  run_zip_test();
  run_reduce_test();
  run_matmul_test();

  if (failures == 0)
  {
    printf("ALL TESTS PASSED\n");
    return 0;
  }
  printf("%d test(s) FAILED\n", failures);
  return 1;
}