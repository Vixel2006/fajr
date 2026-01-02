#include "kernels/squeeze.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <stdarg.h>
#include <string.h>

void squeeze_cpu_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];
  va_list args;
  va_start(args, output);
  u64 axis = va_arg(args, u64);
  va_end(args);

  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;

  u64 new_ndim;
  compute_squeeze_shape_strides(a->shape, a->strides, a->ndim, axis,
                                output->shape, output->strides, &new_ndim);
  output->ndim = new_ndim;
}

void squeeze_cpu_backward(Tensor **inputs, const Tensor *output, ...) {
  // NOTE: No explicit backward operation needed for squeeze, as output->grad
  // points directly to input->grad. Gradients accumulated into output->grad
  // will directly affect input->grad.
}
