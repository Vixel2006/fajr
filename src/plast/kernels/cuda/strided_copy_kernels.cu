#include "plast/kernels/cuda/strided_copy_kernels.h"
#include "plast/kernels/cuda/cuda_kernel_utils.h" // For PLAST_CUDA_CHECK and other utilities
#include "plast/core/types.h" // For PLAST_MAX_DIMS
#include "plast/core/data_buffer.h" // For PLAST_CUDA_CHECK

// Helper to convert a linear index to multi-dimensional coordinates
__device__ void linear_idx_to_coords(size_t linear_idx, const size_t* shape, size_t ndim,
                                     size_t* coords)
{
    for (int i = ndim - 1; i >= 0; --i)
    {
        coords[i] = linear_idx % shape[i];
        linear_idx /= shape[i];
    }
}

// Helper to get the physical index in a strided tensor from multi-dimensional coordinates
__device__ size_t coords_to_physical_idx(const size_t* coords, const size_t* strides, size_t ndim)
{
    size_t physical_idx = 0;
    for (size_t i = 0; i < ndim; ++i)
    {
        physical_idx += coords[i] * strides[i];
    }
    return physical_idx;
}

template <typename T>
__global__ void plast_cuda_strided_copy_kernel(const T* input_data, const size_t* input_shape,
                                               const size_t* input_strides, size_t input_ndim,
                                               T* output_data, const size_t* output_shape,
                                               size_t output_ndim, size_t total_output_elements)
{
    size_t out_flat_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_flat_idx < total_output_elements)
    {
        // Calculate multi-dimensional coordinates for the output tensor
        // Note: output_ndim can be different from input_ndim, but the logical mapping
        // is based on the output's linear index.
        size_t output_coords[PLAST_MAX_DIMS];
        linear_idx_to_coords(out_flat_idx, output_shape, output_ndim, output_coords);

        // Use output_coords to find the corresponding physical index in the input tensor
        // This assumes that the output_coords directly map to the input's logical structure
        // for strided access.
        size_t input_physical_idx =
            coords_to_physical_idx(output_coords, input_strides, input_ndim);

        output_data[out_flat_idx] = input_data[input_physical_idx];
    }
}

// Explicit instantiations for supported types
extern "C" {

void plast_cuda_strided_copy_float(const float* input_data, const size_t* input_shape,
                                   const size_t* input_strides, size_t input_ndim,
                                   float* output_data, const size_t* output_shape,
                                   size_t output_ndim)
{
    size_t total_output_elements = 1;
    for (size_t i = 0; i < output_ndim; ++i)
    {
        total_output_elements *= output_shape[i];
    }

    if (total_output_elements == 0)
    {
        return;
    }

    // Determine grid and block dimensions
    const int BLOCK_SIZE = 256;
    int num_blocks = (total_output_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    plast_cuda_strided_copy_kernel<float><<<num_blocks, BLOCK_SIZE>>>(
        input_data, input_shape, input_strides, input_ndim, output_data, output_shape, output_ndim,
        total_output_elements);
    PLAST_CUDA_CHECK(cudaGetLastError());
}

void plast_cuda_strided_copy_int32(const int32_t* input_data, const size_t* input_shape,
                                   const size_t* input_strides, size_t input_ndim,
                                   int32_t* output_data, const size_t* output_shape,
                                   size_t output_ndim)
{
    size_t total_output_elements = 1;
    for (size_t i = 0; i < output_ndim; ++i)
    {
        total_output_elements *= output_shape[i];
    }

    if (total_output_elements == 0)
    {
        return;
    }

    // Determine grid and block dimensions
    const int BLOCK_SIZE = 256;
    int num_blocks = (total_output_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    plast_cuda_strided_copy_kernel<int32_t><<<num_blocks, BLOCK_SIZE>>>(
        input_data, input_shape, input_strides, input_ndim, output_data, output_shape, output_ndim,
        total_output_elements);
    PLAST_CUDA_CHECK(cudaGetLastError());
}

// Add implementations for other types as needed
} // extern "C"
