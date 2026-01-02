#pragma once

#include "op.h"
#include "tensor.h"
#include <immintrin.h>

#define SIMD_WIDTH 8

void add_cpu_forward(const Tensor **inputs, Tensor *output, ...);
void add_cpu_backward(Tensor **inputs, const Tensor *output, ...);

#ifdef __cplusplus
extern "C" {
#endif

void add_cuda_forward(const Tensor **inputs, Tensor *output, ...);
void add_cuda_backward(Tensor **inputs, const Tensor *output, ...);

#ifdef __cplusplus
}
#endif
