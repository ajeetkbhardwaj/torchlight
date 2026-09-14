// combine.h -- shared definitions + host-side prototypes for the Torchlight
// CUDA kernels, consumed both by the `.cu` implementation and by tests.
//
// Every wrapper below is exactly the signature that the Python ctypes
// interface (`torchlight.backends.cuda_kernel_ops`) binds to.

#ifndef TORCHLIGHT_COMBINE_H
#define TORCHLIGHT_COMBINE_H

#define BLOCK_DIM 1024
#define MAX_DIMS 10
#define TILE 32

#define ADD_FUNC   1
#define MUL_FUNC   2
#define ID_FUNC    3
#define NEG_FUNC   4
#define LT_FUNC    5
#define EQ_FUNC    6
#define SIGMOID_FUNC 7
#define RELU_FUNC  8
#define RELU_BACK_FUNC 9
#define LOG_FUNC   10
#define LOG_BACK_FUNC 11
#define EXP_FUNC   12
#define INV_FUNC   13
#define INV_BACK_FUNC 14
#define IS_CLOSE_FUNC 15
#define MAX_FUNC   16
#define POW        17
#define TANH       18
// Additional ops used by torchlight's operator set (beyond MiniTorch's map).
#define SQRT_FUNC  19
#define ABS_FUNC   20
#define SUB_FUNC   21
#define GT_FUNC    22
#define SIGMOID_BACK_FUNC 23
#define MIN_FUNC   24
#define DIV_FUNC   25

#ifdef __cplusplus
extern "C"
{
#endif

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
      int fn_id);

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
      int fn_id);

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
      int fn_id);

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
      int batch, int m, int p);

#ifdef __cplusplus
}
#endif

#endif // TORCHLIGHT_COMBINE_H