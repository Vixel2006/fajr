#pragma once

#include "definitions.h"
#include "tensor.h"

__device__ inline u64 cuda_get_offset(const u64 *coords, const u64 *strides,
                                      u64 ndim) {
  u64 offset = 0;
  for (u64 i = 0; i < ndim; ++i) {
    offset += coords[i] * strides[i];
  }
  return offset;
}

__device__ __forceinline__ u64 cuda_numel_from_shape(const u64 *shape, u64 ndim) {
  u64 numel = 1;
  for (u64 i = 0; i < ndim; ++i) {
    numel *= shape[i];
  }
  return numel;
}

__device__ inline void cuda_linear_to_coords(u64 linear_idx, const u64 *shape,
                                             u64 ndim, u64 *coords) {
  for (u64 i = ndim; i-- > 0;) {
    coords[i] = linear_idx % shape[i];
    linear_idx /= shape[i];
  }
}

