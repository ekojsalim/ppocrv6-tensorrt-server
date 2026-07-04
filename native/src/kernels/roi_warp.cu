#include "ppocrv6_native/kernels/roi_warp.h"

#include "ppocrv6_native/common/cuda_check.h"

#include <cuda_runtime.h>

namespace ppocrv6_native::kernels {
namespace {

__global__ __launch_bounds__(256)
void roi_warp_normalize_kernel(
    const unsigned char *__restrict__ src, int src_h, int src_w, int src_step,
    const float *__restrict__ m_invs, const int *__restrict__ crop_widths,
    float *__restrict__ dst, int batch_size, int dst_h, int dst_w,
    int output_bgr) {
  __shared__ float s_m[9];
  __shared__ int s_crop_w;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int b = blockIdx.z;
  if (tid < 9) {
    s_m[tid] = m_invs[b * 9 + tid];
  }
  if (tid == 0) {
    s_crop_w = crop_widths[b];
  }
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (b >= batch_size || x >= dst_w || y >= dst_h) {
    return;
  }

  const int plane_size = dst_h * dst_w;
  const int batch_offset = b * 3 * plane_size;
  const int pixel_idx = y * dst_w + x;

  if (x >= s_crop_w) {
    dst[batch_offset + pixel_idx] = 0.0f;
    dst[batch_offset + plane_size + pixel_idx] = 0.0f;
    dst[batch_offset + 2 * plane_size + pixel_idx] = 0.0f;
    return;
  }

  float denom = s_m[6] * x + s_m[7] * y + s_m[8];
  if (fabsf(denom) < 1e-7f) {
    denom = (denom < 0.0f) ? -1e-7f : 1e-7f;
  }
  const float inv_denom = 1.0f / denom;
  float src_x = (s_m[0] * x + s_m[1] * y + s_m[2]) * inv_denom;
  float src_y = (s_m[3] * x + s_m[4] * y + s_m[5]) * inv_denom;

  if (!isfinite(src_x) || !isfinite(src_y)) {
    dst[batch_offset + pixel_idx] = 0.0f;
    dst[batch_offset + plane_size + pixel_idx] = 0.0f;
    dst[batch_offset + 2 * plane_size + pixel_idx] = 0.0f;
    return;
  }

  src_x = fminf(fmaxf(src_x, -1.0f), static_cast<float>(src_w + 1));
  src_y = fminf(fmaxf(src_y, -1.0f), static_cast<float>(src_h + 1));

  int x0 = static_cast<int>(floorf(src_x));
  int y0 = static_cast<int>(floorf(src_y));
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);

  x0 = min(max(x0, 0), src_w - 1);
  x1 = min(max(x1, 0), src_w - 1);
  y0 = min(max(y0, 0), src_h - 1);
  y1 = min(max(y1, 0), src_h - 1);

  const unsigned char *p00 = src + y0 * src_step + x0 * 3;
  const unsigned char *p10 = src + y0 * src_step + x1 * 3;
  const unsigned char *p01 = src + y1 * src_step + x0 * 3;
  const unsigned char *p11 = src + y1 * src_step + x1 * 3;

  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w10 = fx * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w11 = fx * fy;

  const float r = w00 * static_cast<float>(__ldg(p00 + 0)) +
                  w10 * static_cast<float>(__ldg(p10 + 0)) +
                  w01 * static_cast<float>(__ldg(p01 + 0)) +
                  w11 * static_cast<float>(__ldg(p11 + 0));
  const float g = w00 * static_cast<float>(__ldg(p00 + 1)) +
                  w10 * static_cast<float>(__ldg(p10 + 1)) +
                  w01 * static_cast<float>(__ldg(p01 + 1)) +
                  w11 * static_cast<float>(__ldg(p11 + 1));
  const float bl = w00 * static_cast<float>(__ldg(p00 + 2)) +
                   w10 * static_cast<float>(__ldg(p10 + 2)) +
                   w01 * static_cast<float>(__ldg(p01 + 2)) +
                   w11 * static_cast<float>(__ldg(p11 + 2));

  constexpr float inv_127_5 = 1.0f / 127.5f;
  if (output_bgr != 0) {
    dst[batch_offset + pixel_idx] = bl * inv_127_5 - 1.0f;
    dst[batch_offset + plane_size + pixel_idx] = g * inv_127_5 - 1.0f;
    dst[batch_offset + 2 * plane_size + pixel_idx] = r * inv_127_5 - 1.0f;
  } else {
    dst[batch_offset + pixel_idx] = r * inv_127_5 - 1.0f;
    dst[batch_offset + plane_size + pixel_idx] = g * inv_127_5 - 1.0f;
    dst[batch_offset + 2 * plane_size + pixel_idx] = bl * inv_127_5 - 1.0f;
  }
}

} // namespace

void cuda_batch_roi_warp(const GpuImage &src, const float *d_m_invs,
                         const int *d_crop_widths, float *d_dst_batch,
                         int batch_size, int dst_h, int dst_w,
                         ColorOrder output_order, cudaStream_t stream) {
  if (batch_size <= 0) {
    return;
  }

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y, batch_size);

  roi_warp_normalize_kernel<<<grid, block, 0, stream>>>(
      static_cast<const unsigned char *>(src.data), src.rows, src.cols,
      static_cast<int>(src.step), d_m_invs, d_crop_widths, d_dst_batch,
      batch_size, dst_h, dst_w,
      output_order == ColorOrder::kBgr ? 1 : 0);
  PPOCRV6_CUDA_CHECK(cudaGetLastError());
}

} // namespace ppocrv6_native::kernels
