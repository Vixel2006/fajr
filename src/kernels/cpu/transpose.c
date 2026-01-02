#include "kernels/transpose.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <stdarg.h>
#include <string.h>

void transpose_cpu_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];
  va_list args;
  va_start(args, output);
  u64 axis1 = va_arg(args, u64);
  u64 axis2 = va_arg(args, u64);
  va_end(args);

  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;

  output->ndim = a->ndim;
  memcpy(output->shape, a->shape, a->ndim * sizeof(u64));
  memcpy(output->strides, a->strides, a->ndim * sizeof(u64));

  u64 temp_shape = output->shape[axis1];
  output->shape[axis1] = output->shape[axis2];
  output->shape[axis2] = temp_shape;

  u64 temp_strides = output->strides[axis1];
  output->strides[axis1] = output->strides[axis2];
  output->strides[axis2] = temp_strides;
}

void transpose_cpu_backward(Tensor **inputs, const Tensor *output, ...) {
  // NOTE: No explicit backward operation needed for transpose, as output->grad
  // points directly to input->grad. Gradients accumulated into output->grad
  // will directly affect input->grad.
}
