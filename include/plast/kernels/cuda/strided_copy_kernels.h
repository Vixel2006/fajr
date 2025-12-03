#ifndef PLAST_KERNELS_CUDA_STRIDED_COPY_KERNELS_H
#define PLAST_KERNELS_CUDA_STRIDED_COPY_KERNELS_H

#include <stddef.h> // For size_t

#ifdef __cplusplus
extern "C" {
#endif

void plast_cuda_strided_copy_float(const float* input_data, const size_t* input_shape,
                                   const size_t* input_strides, size_t input_ndim,
                                   float* output_data, const size_t* output_shape,
                                   size_t output_ndim);

void plast_cuda_strided_copy_int32(const int32_t* input_data, const size_t* input_shape,
                                   const size_t* input_strides, size_t input_ndim,
                                   int32_t* output_data, const size_t* output_shape,
                                   size_t output_ndim);

// Add declarations for other types as needed (e.g., int64_t, double)

#ifdef __cplusplus
}
#endif

#endif // PLAST_KERNELS_CUDA_STRIDED_COPY_KERNELS_H
