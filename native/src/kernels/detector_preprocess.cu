#include "ppocrv6_native/kernels/detector_preprocess.h"

#include "ppocrv6_native/common/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ppocrv6_native::kernels {
namespace {

__device__ float bilinear_channel(const unsigned char *src, int src_h,
                                  int src_w, int src_step, float src_x,
                                  float src_y, int channel) {
  const int x0_unclamped = static_cast<int>(floorf(src_x));
  const int y0_unclamped = static_cast<int>(floorf(src_y));
  const float wx = src_x - static_cast<float>(x0_unclamped);
  const float wy = src_y - static_cast<float>(y0_unclamped);
  const int x0 = min(max(x0_unclamped, 0), src_w - 1);
  const int x1 = min(max(x0_unclamped + 1, 0), src_w - 1);
  const int y0 = min(max(y0_unclamped, 0), src_h - 1);
  const int y1 = min(max(y0_unclamped + 1, 0), src_h - 1);

  const auto *p00 = src + y0 * src_step + x0 * 3;
  const auto *p10 = src + y0 * src_step + x1 * 3;
  const auto *p01 = src + y1 * src_step + x0 * 3;
  const auto *p11 = src + y1 * src_step + x1 * 3;
  const float top = static_cast<float>(p00[channel]) * (1.0f - wx) +
                    static_cast<float>(p10[channel]) * wx;
  const float bottom = static_cast<float>(p01[channel]) * (1.0f - wx) +
                       static_cast<float>(p11[channel]) * wx;
  return top * (1.0f - wy) + bottom * wy;
}

__global__ void resize_normalize_detector_kernel(
    const unsigned char *__restrict__ src, int src_h, int src_w, int src_step,
    float *__restrict__ dst, int dst_h, int dst_w, int content_h, int content_w, int source_is_bgr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dst_w || y >= dst_h) {
    return;
  }

  const float scale_x =
      static_cast<float>(src_w) / static_cast<float>(content_w);
  const float scale_y =
      static_cast<float>(src_h) / static_cast<float>(content_h);
  const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
  const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

  const int b_channel = source_is_bgr != 0 ? 0 : 2;
  const int g_channel = 1;
  const int r_channel = source_is_bgr != 0 ? 2 : 0;
  const float b = (x >= content_w || y >= content_h) ? 255.0f : bilinear_channel(src, src_h, src_w, src_step, src_x, src_y,
                                   b_channel);
  const float g = (x >= content_w || y >= content_h) ? 255.0f : bilinear_channel(src, src_h, src_w, src_step, src_x, src_y,
                                   g_channel);
  const float r = (x >= content_w || y >= content_h) ? 255.0f : bilinear_channel(src, src_h, src_w, src_step, src_x, src_y,
                                   r_channel);

  constexpr float inv255 = 1.0f / 255.0f;
  constexpr float mean_b = 0.485f;
  constexpr float mean_g = 0.456f;
  constexpr float mean_r = 0.406f;
  constexpr float std_b = 0.229f;
  constexpr float std_g = 0.224f;
  constexpr float std_r = 0.225f;
  const int index = y * dst_w + x;
  const int plane = dst_h * dst_w;
  dst[index] = (b * inv255 - mean_b) / std_b;
  dst[plane + index] = (g * inv255 - mean_g) / std_g;
  dst[2 * plane + index] = (r * inv255 - mean_r) / std_r;
}

} // namespace

void cuda_resize_normalize_detector(const GpuImage &src, float *dst_nchw,
                                    int dst_h, int dst_w, bool source_is_bgr,
                                    cudaStream_t stream) {
  cuda_resize_normalize_detector_padded(src, dst_nchw, dst_h, dst_w,
                                         dst_h, dst_w, source_is_bgr, stream);
}

void cuda_resize_normalize_detector_padded(
    const GpuImage &src, float *dst_nchw, int dst_h, int dst_w,
    int content_h, int content_w, bool source_is_bgr, cudaStream_t stream) {
  if (content_h <= 0 || content_w <= 0 || content_h > dst_h || content_w > dst_w ||
      src.data == nullptr || src.rows <= 0 || src.cols <= 0 ||
      src.step < static_cast<std::size_t>(src.cols * 3) ||
      dst_nchw == nullptr || dst_h <= 0 || dst_w <= 0) {
    throw std::invalid_argument("invalid detector preprocess arguments");
  }
  const dim3 block(32, 8);
  const dim3 grid((dst_w + block.x - 1) / block.x,
                  (dst_h + block.y - 1) / block.y);
  resize_normalize_detector_kernel<<<grid, block, 0, stream>>>(
      static_cast<const unsigned char *>(src.data), src.rows, src.cols,
      static_cast<int>(src.step), dst_nchw, dst_h, dst_w, content_h, content_w,
      source_is_bgr ? 1 : 0);
  PPOCRV6_CUDA_CHECK(cudaGetLastError());
}

} // namespace ppocrv6_native::kernels
