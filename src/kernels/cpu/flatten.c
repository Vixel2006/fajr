#include "kernels/flatten.h"
#include "kernels/cpu_utils.h"
#include "tensor.h"
#include <stdarg.h>
#include <string.h>

void flatten_cpu_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];

  output->data = a->data;
  output->dtype = a->dtype;
  output->device = a->device;
  output->requires_grad = a->requires_grad;
  output->grad = a->grad;

  output->ndim = 2; // Flatten always results in a 2D tensor

  if (a->ndim == 0) { // Scalar
    output->shape[0] = 1;
    output->shape[1] = 1;
  } else if (a->ndim == 1) { // 1D tensor
    output->shape[0] = 1;
    output->shape[1] = a->shape[0];
  } else { // N-D tensor (N > 1)
    output->shape[0] = a->shape[0];
    u64 flattened_dim = 1;
    for (u64 i = 1; i < a->ndim; ++i) {
      flattened_dim *= a->shape[i];
    }
    output->shape[1] = flattened_dim;
  }

  compute_view_strides(a->shape, a->strides, a->ndim, output->shape,
                       output->ndim, output->strides);
}

void flatten_cpu_backward(Tensor **inputs, const Tensor *output, ...) {
  // NOTE: No explicit backward operation needed for flatten, as output->grad
  // points directly to input->grad. Gradients accumulated into output->grad
  // will directly affect input->grad.
}
