#include "plast/core/device_management.h"
#include "plast/core/data_buffer.h" // Added for DataBuffer::fill
#include "plast/ops/init/init_ops.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
#include <cuda_runtime.h>
// Declare CUDA kernels for initialization
extern "C" void plast_cuda_randn_float(float* out, size_t num_elements, int seed);
extern "C" void plast_cuda_uniform_float(float* out, size_t num_elements, float low, float high);
#endif

namespace plast
{
namespace ops
{
namespace init
{

std::shared_ptr<plast::tensor::Tensor>
zeros(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    // Use DataBuffer::fill to set all bytes to 0
    if (device == plast::core::DeviceType::CPU)
    {
        std::memset(output->data(), 0, output->nbytes());
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        PLAST_CUDA_CHECK(cudaMemset(output->data(), 0, output->nbytes()));
#else
        throw std::runtime_error("CUDA is not enabled. Cannot create zeros tensor on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for zeros.");
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor>
ones(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    size_t num_elements = output->num_elements();
    size_t item_size = plast::tensor::get_dtype_size(dtype); // Assuming get_dtype_size is accessible

    if (dtype == plast::core::DType::UINT8 || dtype == plast::core::DType::INT8)
    {
        if (device == plast::core::DeviceType::CPU)
        {
            std::memset(output->data(), 1, output->nbytes());
        }
        else if (device == plast::core::DeviceType::CUDA)
        {
#ifdef PLAST_CUDA_ENABLED
            PLAST_CUDA_CHECK(cudaMemset(output->data(), 1, output->nbytes()));
#else
            throw std::runtime_error("CUDA is not enabled. Cannot create ones tensor on CUDA device.");
#endif
        }
        else
        {
            throw std::runtime_error("Unsupported device type for ones.");
        }
    }
    else
    {
        // Create a temporary host buffer and fill it with 1s
        std::vector<char> host_data(num_elements * item_size);
        if (dtype == plast::core::DType::FLOAT32)
        {
            std::fill((float*)host_data.data(), (float*)host_data.data() + num_elements, 1.0f);
        }
        else if (dtype == plast::core::DType::FLOAT64)
        {
            std::fill((double*)host_data.data(), (double*)host_data.data() + num_elements, 1.0);
        }
        else if (dtype == plast::core::DType::INT16)
        {
            std::fill((int16_t*)host_data.data(), (int16_t*)host_data.data() + num_elements, 1);
        }
        else if (dtype == plast::core::DType::INT32)
        {
            std::fill((int32_t*)host_data.data(), (int32_t*)host_data.data() + num_elements, 1);
        }
        else if (dtype == plast::core::DType::INT64)
        {
            std::fill((int64_t*)host_data.data(), (int64_t*)host_data.data() + num_elements, 1);
        }
        else if (dtype == plast::core::DType::UINT16)
        {
            std::fill((uint16_t*)host_data.data(), (uint16_t*)host_data.data() + num_elements, 1);
        }
        else if (dtype == plast::core::DType::UINT32)
        {
            std::fill((uint32_t*)host_data.data(), (uint32_t*)host_data.data() + num_elements, 1);
        }
        else if (dtype == plast::core::DType::UINT64)
        {
            std::fill((uint64_t*)host_data.data(), (uint64_t*)host_data.data() + num_elements, 1);
        }
        else if (dtype == plast::core::DType::BOOL)
        {
            std::fill((bool*)host_data.data(), (bool*)host_data.data() + num_elements, true);
        }
        else
        {
            throw std::runtime_error("Unsupported DType for ones.");
        }

        // Copy from host to device
        if (device == plast::core::DeviceType::CPU)
        {
            std::memcpy(output->data(), host_data.data(), num_elements * item_size);
        }
        else if (device == plast::core::DeviceType::CUDA)
        {
#ifdef PLAST_CUDA_ENABLED
            PLAST_CUDA_CHECK(cudaMemcpy(output->data(), host_data.data(), num_elements * item_size, cudaMemcpyHostToDevice));
#else
            throw std::runtime_error("CUDA is not enabled. Cannot create ones tensor on CUDA device.");
#endif
        }
        else
        {
            throw std::runtime_error("Unsupported device type for ones.");
        }
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor> randn(const std::vector<size_t>& shape,
                                             plast::core::DType dtype,
                                             plast::core::DeviceType device, int seed)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    // Dispatch to CPU or CUDA kernel
    if (device == plast::core::DeviceType::CPU)
    {
        std::mt19937 generator(seed);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        if (dtype == plast::core::DType::FLOAT32)
        {
            float* data_ptr = (float*) output->data();
            for (size_t i = 0; i < output->num_elements(); ++i)
            {
                data_ptr[i] = distribution(generator);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported DType for randn on CPU.");
        }
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        if (dtype == plast::core::DType::FLOAT32)
        {
            plast_cuda_randn_float((float*) output->data(), output->num_elements(), seed);
        }
        else
        {
            throw std::runtime_error("Unsupported DType for randn on CUDA.");
        }
#else
        throw std::runtime_error("CUDA is not enabled. Cannot create randn tensor on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for randn.");
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor> uniform(const std::vector<size_t>& shape,
                                               plast::core::DType dtype,
                                               plast::core::DeviceType device, float low,
                                               float high, int seed)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    // Dispatch to CPU or CUDA kernel
    if (device == plast::core::DeviceType::CPU)
    {
        std::mt19937 generator(seed); // Use provided seed
        std::uniform_real_distribution<float> distribution(low, high);
        if (dtype == plast::core::DType::FLOAT32)
        {
            float* data_ptr = (float*) output->data();
            for (size_t i = 0; i < output->num_elements(); ++i)
            {
                data_ptr[i] = distribution(generator);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported DType for uniform on CPU.");
        }
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        if (dtype == plast::core::DType::FLOAT32)
        {
            plast_cuda_uniform_float((float*) output->data(), output->num_elements(), low, high);
        }
        else
        {
            throw std::runtime_error("Unsupported DType for uniform on CUDA.");
        }
#else
        throw std::runtime_error(
            "CUDA is not enabled. Cannot create uniform tensor on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for uniform.");
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor> from_data(void* data, const std::vector<size_t>& shape,
                                                 plast::core::DType dtype,
                                                 plast::core::DeviceType device)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    size_t num_elements = output->num_elements();
    size_t nbytes = output->nbytes();

    if (device == plast::core::DeviceType::CPU)
    {
        std::memcpy(output->data(), data, nbytes);
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        PLAST_CUDA_CHECK(cudaMemcpy(output->data(), data, nbytes, cudaMemcpyHostToDevice));
#else
        throw std::runtime_error(
            "CUDA is not enabled. Cannot create tensor from data on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for from_data.");
    }
    return output;
}

} // namespace init
} // namespace ops
} // namespace plast
