#include "zero_grad.h"
#include <stddef.h>

void zero_grad_cpu(Tensor *t) {
  if (t == NULL || t->grad == NULL) {
    return;
  }
  for (size_t i = 0; i < numel(t->grad); ++i) {
    ((float *)t->grad->data)[i] = 0.0f;
  }
}
