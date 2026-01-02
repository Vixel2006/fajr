#pragma once

#include "tensor.h"
#include "definitions.h"

void expand_cpu_forward(const Tensor **inputs, Tensor *output, ...);
void expand_cpu_backward(Tensor **inputs, const Tensor *output, ...);
