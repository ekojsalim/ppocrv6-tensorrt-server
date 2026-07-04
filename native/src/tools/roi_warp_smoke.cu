#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/common/perspective.h"
#include "ppocrv6_native/kernels/roi_warp.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

float sample_channel(const std::vector<unsigned char> &src, int src_h,
                     int src_w, int src_step, const std::array<float, 9> &m,
                     int dst_x, int dst_y, int channel) {
  float denom = m[6] * dst_x + m[7] * dst_y + m[8];
  if (std::fabs(denom) < 1e-7f) {
    denom = (denom < 0.0f) ? -1e-7f : 1e-7f;
  }
  const float inv_denom = 1.0f / denom;
  float src_x = (m[0] * dst_x + m[1] * dst_y + m[2]) * inv_denom;
  float src_y = (m[3] * dst_x + m[4] * dst_y + m[5]) * inv_denom;

  if (!std::isfinite(src_x) || !std::isfinite(src_y)) {
    return 0.0f;
  }

  src_x = std::min(std::max(src_x, -1.0f), static_cast<float>(src_w + 1));
  src_y = std::min(std::max(src_y, -1.0f), static_cast<float>(src_h + 1));

  int x0 = static_cast<int>(std::floor(src_x));
  int y0 = static_cast<int>(std::floor(src_y));
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  const float fx = src_x - static_cast<float>(x0);
  const float fy = src_y - static_cast<float>(y0);

  x0 = std::min(std::max(x0, 0), src_w - 1);
  x1 = std::min(std::max(x1, 0), src_w - 1);
  y0 = std::min(std::max(y0, 0), src_h - 1);
  y1 = std::min(std::max(y1, 0), src_h - 1);

  const auto pixel = [&](int x, int y) -> float {
    return static_cast<float>(src[y * src_step + x * 3 + channel]);
  };

  const float w00 = (1.0f - fx) * (1.0f - fy);
  const float w10 = fx * (1.0f - fy);
  const float w01 = (1.0f - fx) * fy;
  const float w11 = fx * fy;
  const float value = w00 * pixel(x0, y0) + w10 * pixel(x1, y0) +
                      w01 * pixel(x0, y1) + w11 * pixel(x1, y1);
  return value / 127.5f - 1.0f;
}

std::vector<float> cpu_reference(const std::vector<unsigned char> &src,
                                 int src_h, int src_w, int src_step,
                                 const std::vector<float> &m_invs,
                                 const std::vector<int> &crop_widths,
                                 int batch_size, int dst_h, int dst_w) {
  const int plane_size = dst_h * dst_w;
  std::vector<float> out(static_cast<std::size_t>(batch_size) * 3 *
                         plane_size);
  for (int b = 0; b < batch_size; ++b) {
    std::array<float, 9> m{};
    std::copy_n(m_invs.begin() + b * 9, 9, m.begin());
    for (int y = 0; y < dst_h; ++y) {
      for (int x = 0; x < dst_w; ++x) {
        const int pixel_idx = y * dst_w + x;
        const int base = b * 3 * plane_size;
        if (x >= crop_widths[b]) {
          out[base + pixel_idx] = 0.0f;
          out[base + plane_size + pixel_idx] = 0.0f;
          out[base + 2 * plane_size + pixel_idx] = 0.0f;
          continue;
        }
        out[base + pixel_idx] =
            sample_channel(src, src_h, src_w, src_step, m, x, y, 0);
        out[base + plane_size + pixel_idx] =
            sample_channel(src, src_h, src_w, src_step, m, x, y, 1);
        out[base + 2 * plane_size + pixel_idx] =
            sample_channel(src, src_h, src_w, src_step, m, x, y, 2);
      }
    }
  }
  return out;
}

} // namespace

int main() {
  constexpr int src_h = 6;
  constexpr int src_w = 8;
  constexpr int src_step = src_w * 3;
  constexpr int batch_size = 2;
  constexpr int dst_h = 4;
  constexpr int dst_w = 8;

  std::vector<unsigned char> src(src_h * src_step);
  for (int y = 0; y < src_h; ++y) {
    for (int x = 0; x < src_w; ++x) {
      src[y * src_step + x * 3 + 0] =
          static_cast<unsigned char>(10 + x * 7 + y * 3);
      src[y * src_step + x * 3 + 1] =
          static_cast<unsigned char>(40 + x * 5 + y * 11);
      src[y * src_step + x * 3 + 2] =
          static_cast<unsigned char>(90 + x * 13 + y * 2);
    }
  }

  const std::vector<ppocrv6_native::Box> boxes = {
      ppocrv6_native::Box{
          {1.0f, 1.0f},
          {5.0f, 1.0f},
          {5.0f, 4.0f},
          {1.0f, 4.0f},
      },
      ppocrv6_native::Box{
          {6.0f, 0.0f},
          {7.0f, 0.0f},
          {7.0f, 5.0f},
          {6.0f, 5.0f},
      },
  };

  std::vector<float> h_m_invs;
  std::vector<int> h_crop_widths;
  h_m_invs.reserve(batch_size * 9);
  h_crop_widths.reserve(batch_size);
  for (const auto &box : boxes) {
    const auto transform =
        ppocrv6_native::compute_crop_transform(box, dst_h, dst_w);
    h_m_invs.insert(h_m_invs.end(), transform.m_inv.begin(),
                    transform.m_inv.end());
    h_crop_widths.push_back(transform.crop_width);
  }

  ppocrv6_native::CudaPtr<unsigned char> d_src(src.size());
  ppocrv6_native::CudaPtr<float> d_m_invs(h_m_invs.size());
  ppocrv6_native::CudaPtr<int> d_crop_widths(h_crop_widths.size());
  ppocrv6_native::CudaPtr<float> d_dst(static_cast<std::size_t>(batch_size) *
                                       3 * dst_h * dst_w);

  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_src.get(), src.data(), src.size(),
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_m_invs.get(), h_m_invs.data(),
                                h_m_invs.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_crop_widths.get(), h_crop_widths.data(),
                                h_crop_widths.size() * sizeof(int),
                                cudaMemcpyHostToDevice));

  ppocrv6_native::GpuImage image{d_src.get(), src_step, src_h, src_w};
  ppocrv6_native::kernels::cuda_batch_roi_warp(
      image, d_m_invs.get(), d_crop_widths.get(), d_dst.get(), batch_size,
      dst_h, dst_w, ppocrv6_native::kernels::ColorOrder::kRgb);
  PPOCRV6_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_out(static_cast<std::size_t>(batch_size) * 3 * dst_h *
                             dst_w);
  PPOCRV6_CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_dst.get(),
                                gpu_out.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));

  const auto ref = cpu_reference(src, src_h, src_w, src_step, h_m_invs,
                                 h_crop_widths, batch_size, dst_h, dst_w);
  float max_abs = 0.0f;
  std::size_t max_index = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float diff = std::fabs(ref[i] - gpu_out[i]);
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }

  if (max_abs > 1e-5f) {
    throw std::runtime_error("ROI warp mismatch at index " +
                             std::to_string(max_index) + ": max_abs=" +
                             std::to_string(max_abs));
  }

  std::cout << "roi_warp_smoke: ok max_abs=" << max_abs
            << " crop_widths=" << h_crop_widths[0] << ","
            << h_crop_widths[1] << '\n';
  return 0;
}
