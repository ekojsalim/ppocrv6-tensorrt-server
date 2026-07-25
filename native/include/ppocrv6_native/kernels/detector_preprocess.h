#pragma once

#include "ppocrv6_native/decode/gpu_image.h"

#include <cuda_runtime_api.h>

namespace ppocrv6_native::kernels {

void cuda_resize_normalize_detector(const GpuImage &src, float *dst_nchw,
                                    int dst_h, int dst_w, bool source_is_bgr,
                                    cudaStream_t stream = nullptr);

} // namespace ppocrv6_native::kernels
