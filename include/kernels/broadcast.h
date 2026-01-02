#pragma once

#include "tensor.h"
#include "definitions.h"

void broadcast_cpu_forward(const Tensor **inputs, Tensor *output, ...);
void broadcast_cpu_backward(Tensor **inputs, const Tensor *output, ...);
