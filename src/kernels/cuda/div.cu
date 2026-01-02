#include "kernels/cuda/cuda_utils.cuh"
#include "kernels/div.h"
#include <cuda_runtime.h>
#include <stdarg.h>

__global__ void div_cuda_forward_float_contig_kernel(const float *a,
                                                     const float *b, float *c,
                                                     u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    c[idx] = a[idx] / b[idx];
  }
}

__global__ void div_cuda_backward_float_contig_kernel(const float *dout,
                                                      const float *a,
                                                      const float *b, float *da,
                                                      float *db,
                                                      u64 num_elements) {
  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    if (da)
      da[idx] += dout[idx] / b[idx];
    if (db)
      db[idx] -= dout[idx] * a[idx] / (b[idx] * b[idx]);
  }
}

__global__ void div_cuda_forward_float_non_contig_kernel(
    const float *a_data, const u64 *a_strides, const float *b_data,
    const u64 *b_strides, float *c_data, const u64 *c_strides, const u64 *shape,
    u64 ndim, u64 num_elements) {
  u64 linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx < num_elements) {
    u64 coords[MAX_NDIM];
    cuda_linear_to_coords(linear_idx, shape, ndim, coords);
    u64 a_offset = cuda_get_offset(coords, a_strides, ndim);
    u64 b_offset = cuda_get_offset(coords, b_strides, ndim);
    u64 c_offset = cuda_get_offset(coords, c_strides, ndim);
    c_data[c_offset] = a_data[a_offset] / b_data[b_offset];
  }
}

__global__ void div_cuda_backward_float_non_contig_kernel(
    const float *dout_data, const u64 *dout_strides, const float *a_data,
    const u64 *a_strides, const float *b_data, const u64 *b_strides,
    float *da_data, const u64 *da_strides, float *db_data,
    const u64 *db_strides, const u64 *shape, u64 ndim, u64 num_elements) {
  u64 linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx < num_elements) {
    u64 coords[MAX_NDIM];
    cuda_linear_to_coords(linear_idx, shape, ndim, coords);
    u64 dout_offset = cuda_get_offset(coords, dout_strides, ndim);
    u64 a_offset = cuda_get_offset(coords, a_strides, ndim);
    u64 b_offset = cuda_get_offset(coords, b_strides, ndim);
    u64 da_offset = cuda_get_offset(coords, da_strides, ndim);
    u64 db_offset = cuda_get_offset(coords, db_strides, ndim);

    if (da_data)
      da_data[da_offset] += dout_data[dout_offset] / b_data[b_offset];
    if (db_data)
      db_data[db_offset] -= dout_data[dout_offset] * a_data[a_offset] /
                            (b_data[b_offset] * b_data[b_offset]);
  }
}

void div_cuda_forward(const Tensor **inputs, Tensor *output, ...) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];
  u64 num_elements = numel(a);

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      div_cuda_forward_float_contig_kernel<<<grid_size, block_size>>>(
          (const float *)a->data, (const float *)b->data, (float *)output->data,
          num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      div_cuda_forward_float_non_contig_kernel<<<grid_size, block_size>>>(
          (const float *)a->data, a->strides, (const float *)b->data,
          b->strides, (float *)output->data, output->strides, a->shape, a->ndim,
          num_elements);
      break;
    default:
      break;
    }
  }
  cudaDeviceSynchronize();
}

void div_cuda_backward(Tensor **inputs, const Tensor *output, ...) {
  const Tensor *a = inputs[0];
  const Tensor *b = inputs[1];
  u64 num_elements = numel(a);

  int block_size = 256;
  int grid_size = (num_elements + block_size - 1) / block_size;

  if (is_contiguous(a) && is_contiguous(b) && is_contiguous(output)) {
    switch (a->dtype) {
    case FLOAT32:
      div_cuda_backward_float_contig_kernel<<<grid_size, block_size>>>(
          (const float *)output->grad->data, (const float *)a->data,
          (const float *)b->data,
          a->requires_grad ? (float *)a->grad->data : NULL,
          b->requires_grad ? (float *)b->grad->data : NULL, num_elements);
      break;
    default:
      break;
    }
  } else {
    switch (a->dtype) {
    case FLOAT32:
      div_cuda_backward_float_non_contig_kernel<<<grid_size, block_size>>>(
          (const float *)output->grad->data, output->grad->strides,
          (const float *)a->data, a->strides, (const float *)b->data,
          b->strides, a->requires_grad ? (float *)a->grad->data : NULL,
          a->requires_grad ? a->grad->strides : NULL,
          b->requires_grad ? (float *)b->grad->data : NULL,
          b->requires_grad ? b->grad->strides : NULL, a->shape, a->ndim,
          num_elements);
      break;
    default:
      break;
    }
  }
  cudaDeviceSynchronize();
}
