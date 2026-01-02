#include "kernels/cpu/cpu_tensor_init.h"
#include <string.h>

void zeros_cpu(Tensor *t, u64 num_elements) {
  u64 bytes = num_elements * dtype_size(t->dtype);
  memset(t->data, 0, bytes);
}

void ones_cpu(Tensor *t, u64 num_elements) {
  switch (t->dtype) {
  case INT32: {
    i32 *d = (i32 *)t->data;
    for (u64 i = 0; i < num_elements; ++i)
      d[i] = 1;
    break;
  }
  case FLOAT32: {
    float *d = (float *)t->data;
    for (u64 i = 0; i < num_elements; ++i)
      d[i] = 1.0f;
    break;
  }
  default:
    break;
  }
}

void set_ones_grad_cpu(Tensor *t) {
  u64 num_elements = numel(t);
  switch (t->dtype) {
  case INT32: {
    i32 *d = (i32 *)t->grad->data;
    for (u64 i = 0; i < num_elements; ++i)
      d[i] = 1;
    break;
  }
  case FLOAT32: {
    float *d = (float *)t->grad->data;
    for (u64 i = 0; i < num_elements; ++i)
      d[i] = 1.0f;
    break;
  }
  default:
    break;
  }
}
