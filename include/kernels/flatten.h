#pragma once

#include "tensor.h"
#include "definitions.h"

void flatten_cpu_forward(const Tensor **inputs, Tensor *output, ...);
void flatten_cpu_backward(Tensor **inputs, const Tensor *output, ...);
