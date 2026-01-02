#include "arena.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "kernels/conv2d.h"
#include "kernels/flatten.h"
#include "kernels/matmul.h"
#include "kernels/transpose.h"
#include "op.h"
#include <stdarg.h>
#include <string.h>

__global__ void im2col_cuda_float_kernel(const float *img, float *buffer, u64 N,
                                         u64 C, u64 H_in, u64 W_in, u64 kh,
                                         u64 kw, u64 stride, u64 img_stride_N,
                                         u64 img_stride_C, u64 img_stride_H,
                                         u64 img_stride_W) {
  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  u64 output_elements = N * H_out * W_out * C * kh * kw;

  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= output_elements) {
    return;
  }

  u64 col_in_patch = idx % (kh * kw);
  u64 kr = col_in_patch / kw;
  u64 kc = col_in_patch % kw;

  u64 idx_after_kernel = idx / (kh * kw);
  u64 c = idx_after_kernel % C;

  u64 idx_after_channel = idx_after_kernel / C;
  u64 col_in_output_spatial = idx_after_channel % (H_out * W_out);
  u64 out_h = col_in_output_spatial / W_out;
  u64 out_w = col_in_output_spatial % W_out;

  u64 batch = idx_after_channel / (H_out * W_out);

  u64 in_h = out_h * stride + kr;
  u64 in_w = out_w * stride + kc;

  u64 img_idx = batch * img_stride_N + c * img_stride_C + in_h * img_stride_H +
                in_w * img_stride_W;

  buffer[idx] = img[img_idx];
}

__global__ void col2im_cuda_float_kernel(const float *buffer, float *img, u64 N,
                                         u64 C, u64 H_in, u64 W_in, u64 kh,
                                         u64 kw, u64 stride, u64 img_stride_N,
                                         u64 img_stride_C, u64 img_stride_H,
                                         u64 img_stride_W) {
  u64 H_out = (H_in - kh) / stride + 1;
  u64 W_out = (W_in - kw) / stride + 1;

  u64 input_buffer_elements = N * H_out * W_out * C * kh * kw;

  u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_buffer_elements) {
    return;
  }

  u64 col_in_patch = idx % (kh * kw);
  u64 kr = col_in_patch / kw;
  u64 kc = col_in_patch % kw;

  u64 idx_after_kernel = idx / (kh * kw);
  u64 c = idx_after_kernel % C;

  u64 idx_after_channel = idx_after_kernel / C;
  u64 col_in_output_spatial = idx_after_channel % (H_out * W_out);
  u64 out_h = col_in_output_spatial / W_out;
  u64 out_w = col_in_output_spatial % W_out;

  u64 batch = idx_after_channel / (H_out * W_out);

  u64 in_h = out_h * stride + kr;
  u64 in_w = out_w * stride + kc;

  if (in_h < H_in && in_w < W_in) {
    u64 img_idx = batch * img_stride_N + c * img_stride_C +
                  in_h * img_stride_H + in_w * img_stride_W;

    atomicAdd(&img[img_idx], buffer[idx]);
  }
}

void conv2d_cuda_forward(const Tensor **inputs, Tensor *output, ...) {}

void conv2d_cuda_backward(Tensor **inputs, const Tensor *output, ...) {}
