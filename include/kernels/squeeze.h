#pragma once

#include "tensor.h"
#include "definitions.h"

void squeeze_cpu_forward(const Tensor **inputs, Tensor *output, ...);
void squeeze_cpu_backward(Tensor **inputs, const Tensor *output, ...);
