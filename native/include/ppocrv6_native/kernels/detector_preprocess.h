#pragma once

#include "ppocrv6_native/decode/gpu_image.h"

#include <cuda_runtime_api.h>

namespace ppocrv6_native::kernels {

// Resize into the upper-left content rectangle; fill the remaining canvas white.
void cuda_resize_normalize_detector_padded(
    const GpuImage &src, float *dst_nchw, int dst_h, int dst_w,
    int content_h, int content_w, bool source_is_bgr,
    cudaStream_t stream = nullptr);

void cuda_resize_normalize_detector(const GpuImage &src, float *dst_nchw,
                                    int dst_h, int dst_w, bool source_is_bgr,
                                    cudaStream_t stream = nullptr);

} // namespace ppocrv6_native::kernels
