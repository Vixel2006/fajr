#include "kernels/view.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <stdarg.h>
#include <string.h>

void view_cpu_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];
  va_list args;
  va_start(args, output);
  u64 new_ndim = va_arg(args, u64);
  u64 *new_shape_arg = va_arg(args, u64 *);
  va_end(args);

  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;

  output->ndim = new_ndim;
  memcpy(output->shape, new_shape_arg, new_ndim * sizeof(u64));
  compute_view_strides(a->shape, a->strides, a->ndim, output->shape,
                       output->ndim, output->strides);
}

void view_cpu_backward(Tensor **inputs, const Tensor *output, ...) {
  // NOTE: No explicit backward operation needed for view, as output->grad
  // points directly to input->grad. Gradients accumulated into output->grad
  // will directly affect input->grad.
}
