#include "cuda_utils.cuh"
#include "zero_grad.h"

__global__ void zero_grad_kernel(float *grad, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    grad[idx] = 0.0f;
  }
}

void zero_grad_cuda(Tensor *t) {
  if (t == NULL || t->grad == NULL) {
    return;
  }

  size_t num_elements = numel(t->grad);
  float *grad_data = (float *)t->grad->data;

  int blockSize = 256;
  int numBlocks = (num_elements + blockSize - 1) / blockSize;

  zero_grad_kernel<<<numBlocks, blockSize>>>(grad_data, num_elements);
}
