#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/kernels/roi_warp.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::map<std::string, std::string> parse_args(int argc, char **argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) {
      throw std::runtime_error("unexpected positional argument: " + key);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + key);
    }
    args[key] = argv[++i];
  }
  return args;
}

const std::string &required(const std::map<std::string, std::string> &args,
                            const std::string &key) {
  const auto it = args.find(key);
  if (it == args.end()) {
    throw std::runtime_error("missing required argument " + key);
  }
  return it->second;
}

int required_int(const std::map<std::string, std::string> &args,
                 const std::string &key) {
  return std::stoi(required(args, key));
}

float optional_float(const std::map<std::string, std::string> &args,
                     const std::string &key, float default_value) {
  const auto it = args.find(key);
  if (it == args.end()) {
    return default_value;
  }
  return std::stof(it->second);
}

template <typename T>
std::vector<T> read_binary(const std::string &path, std::size_t expected_count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  const std::streamsize bytes = in.tellg();
  if (bytes < 0) {
    throw std::runtime_error("failed to stat " + path);
  }
  const std::size_t expected_bytes = expected_count * sizeof(T);
  if (static_cast<std::size_t>(bytes) != expected_bytes) {
    throw std::runtime_error("unexpected file size for " + path + ": got " +
                             std::to_string(bytes) + " bytes, expected " +
                             std::to_string(expected_bytes));
  }
  std::vector<T> out(expected_count);
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(expected_bytes));
  if (!in) {
    throw std::runtime_error("failed to read " + path);
  }
  return out;
}

template <typename T>
void write_binary(const std::string &path, const std::vector<T> &values) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output " + path);
  }
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out) {
    throw std::runtime_error("failed to write " + path);
  }
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);

  const int src_h = required_int(args, "--src-h");
  const int src_w = required_int(args, "--src-w");
  const int src_step = required_int(args, "--src-step");
  const int batch_size = required_int(args, "--batch-size");
  const int dst_h = required_int(args, "--dst-h");
  const int dst_w = required_int(args, "--dst-w");
  const float tolerance = optional_float(args, "--tolerance", 1e-5f);

  if (src_h <= 0 || src_w <= 0 || src_step < src_w * 3 || batch_size <= 0 ||
      dst_h <= 0 || dst_w <= 0) {
    throw std::runtime_error("invalid shape arguments");
  }

  const std::size_t src_bytes = static_cast<std::size_t>(src_h) * src_step;
  const std::size_t dst_count =
      static_cast<std::size_t>(batch_size) * 3 * dst_h * dst_w;

  const auto h_src = read_binary<unsigned char>(required(args, "--src"), src_bytes);
  const auto h_m_invs =
      read_binary<float>(required(args, "--m-invs"),
                         static_cast<std::size_t>(batch_size) * 9);
  const auto h_crop_widths =
      read_binary<int>(required(args, "--crop-widths"),
                       static_cast<std::size_t>(batch_size));

  ppocrv6_native::CudaPtr<unsigned char> d_src(h_src.size());
  ppocrv6_native::CudaPtr<float> d_m_invs(h_m_invs.size());
  ppocrv6_native::CudaPtr<int> d_crop_widths(h_crop_widths.size());
  ppocrv6_native::CudaPtr<float> d_dst(dst_count);

  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_src.get(), h_src.data(), h_src.size(),
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_m_invs.get(), h_m_invs.data(),
                                h_m_invs.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_crop_widths.get(), h_crop_widths.data(),
                                h_crop_widths.size() * sizeof(int),
                                cudaMemcpyHostToDevice));

  const auto color_it = args.find("--color-mode");
  const std::string color_mode =
      color_it == args.end() ? std::string("rgb") : color_it->second;
  const auto output_order =
      color_mode == "bgr" ? ppocrv6_native::kernels::ColorOrder::kBgr
                          : ppocrv6_native::kernels::ColorOrder::kRgb;
  if (color_mode != "rgb" && color_mode != "bgr") {
    throw std::runtime_error("--color-mode must be rgb or bgr");
  }

  ppocrv6_native::GpuImage image{d_src.get(),
                                 static_cast<std::size_t>(src_step), src_h,
                                 src_w};
  ppocrv6_native::kernels::cuda_batch_roi_warp(
      image, d_m_invs.get(), d_crop_widths.get(), d_dst.get(), batch_size,
      dst_h, dst_w, output_order);
  PPOCRV6_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_dst(dst_count);
  PPOCRV6_CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst.get(),
                                h_dst.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));

  const auto output_it = args.find("--output");
  if (output_it != args.end()) {
    write_binary(output_it->second, h_dst);
  }

  float max_abs = 0.0f;
  double mean_abs = 0.0;
  std::size_t max_index = 0;
  const auto reference_it = args.find("--reference");
  if (reference_it != args.end()) {
    const auto reference = read_binary<float>(reference_it->second, dst_count);
    for (std::size_t i = 0; i < h_dst.size(); ++i) {
      const float diff = std::fabs(h_dst[i] - reference[i]);
      mean_abs += static_cast<double>(diff);
      if (diff > max_abs) {
        max_abs = diff;
        max_index = i;
      }
    }
    mean_abs /= static_cast<double>(h_dst.size());
    if (max_abs > tolerance) {
      throw std::runtime_error("ROI file parity mismatch: max_abs=" +
                               std::to_string(max_abs) + " tolerance=" +
                               std::to_string(tolerance) +
                               " max_index=" + std::to_string(max_index));
    }
  }

  std::cout << "roi_warp_file_smoke: ok batch_size=" << batch_size
            << " dst=" << dst_h << "x" << dst_w
            << " color_mode=" << color_mode << " max_abs=" << max_abs
            << " mean_abs=" << mean_abs << " max_index=" << max_index
            << '\n';
  return 0;
}
