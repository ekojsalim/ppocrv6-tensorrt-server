#pragma once

#include "ppocrv6_native/decode/gpu_image.h"

#include <cuda_runtime.h>

namespace ppocrv6_native::kernels {

enum class ColorOrder {
  kRgb = 0,
  kBgr = 1,
};

void cuda_batch_roi_warp(const GpuImage &src, const float *d_m_invs,
                         const int *d_crop_widths, float *d_dst_batch,
                         int batch_size, int dst_h, int dst_w,
                         ColorOrder output_order = ColorOrder::kRgb,
                         cudaStream_t stream = nullptr);

} // namespace ppocrv6_native::kernels
