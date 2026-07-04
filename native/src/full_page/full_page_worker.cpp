#include "ppocrv6_native/full_page/full_page_worker.h"

#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/common/perspective.h"
#include "ppocrv6_native/decode/gpu_image.h"
#include "ppocrv6_native/kernels/roi_warp.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
#include <opencv2/imgproc.hpp>
#endif

namespace ppocrv6_native::full_page {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t stop) {
  PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  return elapsed;
}

void append_json_escaped(std::ostringstream &out, const std::string &value) {
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (ch < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(ch) << std::dec << std::setfill(' ');
      } else {
        out << static_cast<char>(ch);
      }
      break;
    }
  }
  out << '"';
}

void append_float(std::ostringstream &out, double value) {
  if (std::isfinite(value)) {
    out << std::setprecision(9) << value;
  } else {
    out << "0.0";
  }
}

void append_int_array(std::ostringstream &out, const std::vector<int> &values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    out << values[i];
  }
  out << ']';
}

void append_float_array(std::ostringstream &out,
                        const std::vector<float> &values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    append_float(out, values[i]);
  }
  out << ']';
}

const char *limit_type_to_string(DetectorLimitType value) {
  switch (value) {
  case DetectorLimitType::kMax:
    return "max";
  case DetectorLimitType::kMin:
    return "min";
  case DetectorLimitType::kResizeLong:
    return "resize_long";
  }
  return "max";
}

struct DetectorInput {
  int height = 0;
  int width = 0;
  std::vector<float> nchw;
};

std::pair<int, int> detector_resize_shape(
    int src_h, int src_w, const DetectorPreprocessConfig &config) {
  if (src_h <= 0 || src_w <= 0 || config.limit_side_len <= 0 ||
      config.max_side_limit <= 0) {
    throw std::runtime_error("invalid detector preprocess shape config");
  }

  double ratio = 1.0;
  switch (config.limit_type) {
  case DetectorLimitType::kMax:
    ratio = std::min(1.0, static_cast<double>(config.limit_side_len) /
                              static_cast<double>(std::max(src_h, src_w)));
    break;
  case DetectorLimitType::kMin:
    ratio = std::max(1.0, static_cast<double>(config.limit_side_len) /
                              static_cast<double>(std::min(src_h, src_w)));
    break;
  case DetectorLimitType::kResizeLong:
    ratio = static_cast<double>(config.limit_side_len) /
            static_cast<double>(std::max(src_h, src_w));
    break;
  }

  int resize_h = static_cast<int>(static_cast<double>(src_h) * ratio);
  int resize_w = static_cast<int>(static_cast<double>(src_w) * ratio);
  if (std::max(resize_h, resize_w) > config.max_side_limit) {
    ratio = static_cast<double>(config.max_side_limit) /
            static_cast<double>(std::max(resize_h, resize_w));
    resize_h = static_cast<int>(static_cast<double>(resize_h) * ratio);
    resize_w = static_cast<int>(static_cast<double>(resize_w) * ratio);
  }

  resize_h =
      std::max(static_cast<int>(std::round(static_cast<double>(resize_h) / 32.0)) *
                   32,
               32);
  resize_w =
      std::max(static_cast<int>(std::round(static_cast<double>(resize_w) / 32.0)) *
                   32,
               32);
  return {resize_h, resize_w};
}

void write_normalized_bgr_nchw(const std::uint8_t *bgr, int height, int width,
                               std::size_t stride,
                               std::vector<float> &out) {
  const std::size_t plane =
      static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  out.assign(plane * 3U, 0.0f);
  constexpr float inv255 = 1.0f / 255.0f;
  constexpr float mean_b = 0.485f;
  constexpr float mean_g = 0.456f;
  constexpr float mean_r = 0.406f;
  constexpr float std_b = 0.229f;
  constexpr float std_g = 0.224f;
  constexpr float std_r = 0.225f;
  for (int y = 0; y < height; ++y) {
    const auto *row =
        bgr + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
    for (int x = 0; x < width; ++x) {
      const std::size_t index =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x);
      const auto *pixel = row + static_cast<std::size_t>(x) * 3U;
      out[index] = (static_cast<float>(pixel[0]) * inv255 - mean_b) / std_b;
      out[plane + index] =
          (static_cast<float>(pixel[1]) * inv255 - mean_g) / std_g;
      out[(2U * plane) + index] =
          (static_cast<float>(pixel[2]) * inv255 - mean_r) / std_r;
    }
  }
}

DetectorInput preprocess_detector_input_opencv(
    const std::uint8_t *image, int image_height, int image_width,
    int image_stride, SourceColorOrder source_color_order,
    const DetectorPreprocessConfig &config) {
  const auto [resize_h, resize_w] =
      detector_resize_shape(image_height, image_width, config);

  DetectorInput out;
  out.height = resize_h;
  out.width = resize_w;

#if defined(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
  cv::Mat source(image_height, image_width, CV_8UC3,
                 const_cast<std::uint8_t *>(image),
                 static_cast<std::size_t>(image_stride));
  cv::Mat bgr;
  if (source_color_order == SourceColorOrder::kRgb) {
    cv::cvtColor(source, bgr, cv::COLOR_RGB2BGR);
  } else {
    bgr = source;
  }

  cv::Mat resized;
  if (resize_h == image_height && resize_w == image_width) {
    resized = bgr;
  } else {
    cv::resize(bgr, resized, cv::Size(resize_w, resize_h), 0.0, 0.0,
               cv::INTER_LINEAR);
  }
  write_normalized_bgr_nchw(resized.data, resize_h, resize_w, resized.step,
                            out.nchw);
#else
  const std::size_t packed_bytes =
      static_cast<std::size_t>(resize_h) * static_cast<std::size_t>(resize_w) *
      3U;
  std::vector<std::uint8_t> resized_bgr(packed_bytes);
  const float scale_x = static_cast<float>(image_width) /
                        static_cast<float>(std::max(resize_w, 1));
  const float scale_y = static_cast<float>(image_height) /
                        static_cast<float>(std::max(resize_h, 1));
  for (int y = 0; y < resize_h; ++y) {
    const float src_y = std::max((static_cast<float>(y) + 0.5f) * scale_y -
                                     0.5f,
                                 0.0f);
    const int y0 = std::min(static_cast<int>(std::floor(src_y)), image_height - 1);
    const int y1 = std::min(y0 + 1, image_height - 1);
    const float wy = src_y - static_cast<float>(y0);
    for (int x = 0; x < resize_w; ++x) {
      const float src_x =
          std::max((static_cast<float>(x) + 0.5f) * scale_x - 0.5f, 0.0f);
      const int x0 =
          std::min(static_cast<int>(std::floor(src_x)), image_width - 1);
      const int x1 = std::min(x0 + 1, image_width - 1);
      const float wx = src_x - static_cast<float>(x0);
      float values[3] = {0.0f, 0.0f, 0.0f};
      const int ys[2] = {y0, y1};
      const int xs[2] = {x0, x1};
      for (int yy = 0; yy < 2; ++yy) {
        const float wy_factor = yy == 0 ? (1.0f - wy) : wy;
        const auto *row = image + static_cast<std::size_t>(ys[yy]) *
                                      static_cast<std::size_t>(image_stride);
        for (int xx = 0; xx < 2; ++xx) {
          const float weight = wy_factor * (xx == 0 ? (1.0f - wx) : wx);
          const auto *pixel = row + static_cast<std::size_t>(xs[xx]) * 3U;
          if (source_color_order == SourceColorOrder::kRgb) {
            values[0] += static_cast<float>(pixel[2]) * weight;
            values[1] += static_cast<float>(pixel[1]) * weight;
            values[2] += static_cast<float>(pixel[0]) * weight;
          } else {
            values[0] += static_cast<float>(pixel[0]) * weight;
            values[1] += static_cast<float>(pixel[1]) * weight;
            values[2] += static_cast<float>(pixel[2]) * weight;
          }
        }
      }
      auto *dst = resized_bgr.data() +
                  (static_cast<std::size_t>(y) *
                       static_cast<std::size_t>(resize_w) +
                   static_cast<std::size_t>(x)) *
                      3U;
      for (int c = 0; c < 3; ++c) {
        dst[c] = static_cast<std::uint8_t>(
            std::clamp(std::round(values[c]), 0.0f, 255.0f));
      }
    }
  }
  write_normalized_bgr_nchw(
      resized_bgr.data(), resize_h, resize_w,
      static_cast<std::size_t>(resize_w) * 3U, out.nchw);
#endif
  return out;
}

int max_bucket(const std::vector<int> &buckets) {
  if (buckets.empty()) {
    throw std::runtime_error("recognition buckets must not be empty");
  }
  return *std::max_element(buckets.begin(), buckets.end());
}

int assign_bucket(int width, const std::vector<int> &buckets) {
  for (const int bucket : buckets) {
    if (width <= bucket) {
      return bucket;
    }
  }
  return buckets.back();
}

ppocrv6_native::kernels::ColorOrder
roi_color_order(SourceColorOrder source_color_order) {
  return source_color_order == SourceColorOrder::kBgr
             ? ppocrv6_native::kernels::ColorOrder::kBgr
             : ppocrv6_native::kernels::ColorOrder::kRgb;
}

struct PendingCrop {
  int line_index = 0;
  CropTransform transform;
};

} // namespace

class FullPageWorker::Impl {
public:
  explicit Impl(FullPageWorkerConfig config)
      : config_(std::move(config)), detector_(config_.detector),
        recognizer_(config_.recognizer) {
    if (config_.recognition_height != 48) {
      throw std::runtime_error("recognition_height must be 48 for this engine");
    }
    if (config_.recognizer.max_width <= 0 ||
        config_.recognizer.max_batch_size <= 0) {
      throw std::runtime_error("invalid full-page recognizer config");
    }
    std::sort(config_.recognition_buckets.begin(),
              config_.recognition_buckets.end());
    config_.recognition_buckets.erase(
        std::unique(config_.recognition_buckets.begin(),
                    config_.recognition_buckets.end()),
        config_.recognition_buckets.end());
    for (const int bucket : config_.recognition_buckets) {
      if (bucket <= 0 || bucket > config_.recognizer.max_width) {
        throw std::runtime_error("recognition bucket is outside worker limits");
      }
    }

    max_bucket_width_ = max_bucket(config_.recognition_buckets);
    const std::size_t transform_count =
        static_cast<std::size_t>(config_.recognizer.max_batch_size) * 9U;
    const std::size_t roi_count =
        static_cast<std::size_t>(config_.recognizer.max_batch_size) * 3U *
        static_cast<std::size_t>(config_.recognition_height) *
        static_cast<std::size_t>(max_bucket_width_);
    d_m_invs_.reset(transform_count);
    d_crop_widths_.reset(static_cast<std::size_t>(
        config_.recognizer.max_batch_size));
    d_roi_batch_.reset(roi_count);
    PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream_));
  }

  ~Impl() {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  FullPageResult recognize_page(const std::uint8_t *image, int image_height,
                                int image_width, int image_stride,
                                SourceColorOrder source_color_order,
                                const float *detector_nchw,
                                int detector_height, int detector_width) {
    if (image == nullptr) {
      throw std::runtime_error("source image pointer is null");
    }
    if (detector_nchw == nullptr) {
      throw std::runtime_error("detector tensor pointer is null");
    }
    if (image_height <= 0 || image_width <= 0 ||
        image_stride < image_width * 3 || detector_height <= 0 ||
        detector_width <= 0) {
      throw std::runtime_error("invalid full-page input shape");
    }

    std::lock_guard<std::mutex> guard(mutex_);
    const auto total_start = std::chrono::steady_clock::now();

    FullPageResult result;
    result.source_height = image_height;
    result.source_width = image_width;
    result.detector_height = detector_height;
    result.detector_width = detector_width;
    return run_pipeline_locked(image, image_height, image_width, image_stride,
                               source_color_order, detector_nchw, result,
                               total_start);
  }

  FullPageResult recognize_page_image(const std::uint8_t *image,
                                      int image_height, int image_width,
                                      int image_stride,
                                      SourceColorOrder source_color_order) {
    if (image == nullptr) {
      throw std::runtime_error("source image pointer is null");
    }
    if (image_height <= 0 || image_width <= 0 ||
        image_stride < image_width * 3) {
      throw std::runtime_error("invalid full-page image shape");
    }

    std::lock_guard<std::mutex> guard(mutex_);
    const auto total_start = std::chrono::steady_clock::now();
    const auto preprocess_start = std::chrono::steady_clock::now();
    auto detector_input = preprocess_detector_input_opencv(
        image, image_height, image_width, image_stride, source_color_order,
        config_.detector_preprocess);

    FullPageResult result;
    result.source_height = image_height;
    result.source_width = image_width;
    result.detector_height = detector_input.height;
    result.detector_width = detector_input.width;
    result.timing.detector_preprocess_ms =
        elapsed_ms(preprocess_start, std::chrono::steady_clock::now());
    return run_pipeline_locked(image, image_height, image_width, image_stride,
                               source_color_order, detector_input.nchw.data(),
                               result, total_start);
  }

  FullPageResult run_pipeline_locked(
      const std::uint8_t *image, int image_height, int image_width,
      int image_stride, SourceColorOrder source_color_order,
      const float *detector_nchw, FullPageResult result,
      std::chrono::steady_clock::time_point total_start) {

    auto detection_result =
        detector_.detect_f32(detector_nchw, 1, result.detector_height,
                             result.detector_width, true);
    result.detector_output_height = detection_result.output_h;
    result.detector_output_width = detection_result.output_w;
    result.timing.detect_ms = detection_result.elapsed_ms;
    result.timing.detector_h2d_ms = detection_result.h2d_ms;
    result.timing.detector_execute_ms = detection_result.execute_ms;
    result.timing.detector_d2h_ms = detection_result.d2h_ms;
    result.timing.detector_output_bytes = detection_result.output_bytes;

    const auto post_start = std::chrono::steady_clock::now();
    auto boxes = detection::postprocess_db_map(
        detection_result.output.data(), detection_result.output_h,
        detection_result.output_w, image_height, image_width,
        config_.postprocess);
    result.timing.postprocess_ms =
        elapsed_ms(post_start, std::chrono::steady_clock::now());

    result.lines.reserve(boxes.size());
    std::map<int, std::vector<PendingCrop>> groups;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
      auto probe = compute_crop_transform(boxes[i].box,
                                          config_.recognition_height,
                                          max_bucket_width_);
      const int bucket =
          assign_bucket(probe.natural_width, config_.recognition_buckets);
      auto transform =
          compute_crop_transform(boxes[i].box, config_.recognition_height,
                                 bucket);

      FullPageLine line;
      line.index = static_cast<int>(i);
      line.detection = boxes[i];
      line.bucket_width = bucket;
      line.crop_width = transform.crop_width;
      line.natural_width = transform.natural_width;
      line.vertical = transform.vertical;
      line.clipped = transform.natural_width > bucket;
      result.lines.push_back(std::move(line));
      groups[bucket].push_back(PendingCrop{static_cast<int>(i), transform});
    }

    if (result.lines.empty()) {
      result.timing.total_ms =
          elapsed_ms(total_start, std::chrono::steady_clock::now());
      return result;
    }

    copy_source_image(image, image_height, image_width, image_stride);
    GpuImage gpu_image{d_source_image_.get(),
                       static_cast<std::size_t>(image_stride), image_height,
                       image_width};

    std::vector<float> h_m_invs;
    std::vector<int> h_crop_widths;
    const int max_batch = config_.recognizer.max_batch_size;
    h_m_invs.reserve(static_cast<std::size_t>(max_batch) * 9U);
    h_crop_widths.reserve(static_cast<std::size_t>(max_batch));

    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&event_start));
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&event_stop));
    try {
      for (const auto &[bucket, crops] : groups) {
        for (std::size_t start = 0; start < crops.size();
             start += static_cast<std::size_t>(max_batch)) {
          const int chunk_count = static_cast<int>(std::min<std::size_t>(
              static_cast<std::size_t>(max_batch), crops.size() - start));
          h_m_invs.clear();
          h_crop_widths.clear();
          for (int row = 0; row < chunk_count; ++row) {
            const auto &transform = crops[start + static_cast<std::size_t>(row)]
                                        .transform;
            h_m_invs.insert(h_m_invs.end(), transform.m_inv.begin(),
                            transform.m_inv.end());
            h_crop_widths.push_back(transform.crop_width);
          }

          PPOCRV6_CUDA_CHECK(cudaEventRecord(event_start, stream_));
          PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(
              d_m_invs_.get(), h_m_invs.data(),
              h_m_invs.size() * sizeof(float), cudaMemcpyHostToDevice,
              stream_));
          PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(
              d_crop_widths_.get(), h_crop_widths.data(),
              h_crop_widths.size() * sizeof(int), cudaMemcpyHostToDevice,
              stream_));
          kernels::cuda_batch_roi_warp(
              gpu_image, d_m_invs_.get(), d_crop_widths_.get(),
              d_roi_batch_.get(), chunk_count, config_.recognition_height,
              bucket, roi_color_order(source_color_order), stream_);
          PPOCRV6_CUDA_CHECK(cudaEventRecord(event_stop, stream_));
          result.timing.roi_ms += elapsed_event_ms(event_start, event_stop);

          auto recog_result = recognizer_.recognize_device_f32(
              d_roi_batch_.get(), chunk_count, bucket, chunk_count, false);
          result.timing.recognize_ms += recog_result.elapsed_ms;
          result.timing.recognition_output_bytes += recog_result.output_bytes;
          result.recognition_chunks.insert(result.recognition_chunks.end(),
                                           recog_result.chunks.begin(),
                                           recog_result.chunks.end());
          for (int row = 0; row < chunk_count; ++row) {
            const int line_index =
                crops[start + static_cast<std::size_t>(row)].line_index;
            result.lines[static_cast<std::size_t>(line_index)].prediction =
                std::move(recog_result.predictions[static_cast<std::size_t>(row)]);
          }
        }
      }
    } catch (...) {
      cudaEventDestroy(event_start);
      cudaEventDestroy(event_stop);
      throw;
    }
    PPOCRV6_CUDA_CHECK(cudaEventDestroy(event_start));
    PPOCRV6_CUDA_CHECK(cudaEventDestroy(event_stop));

    result.timing.total_ms =
        elapsed_ms(total_start, std::chrono::steady_clock::now());
    return result;
  }

  [[nodiscard]] std::string info_json() const {
    std::ostringstream out;
    out << '{';
    out << "\"detector\":" << detector_.info_json();
    out << ",\"recognizer\":" << recognizer_.info_json();
    out << ",\"recognition_height\":" << config_.recognition_height;
    out << ",\"recognition_buckets\":[";
    for (std::size_t i = 0; i < config_.recognition_buckets.size(); ++i) {
      if (i != 0) {
        out << ',';
      }
      out << config_.recognition_buckets[i];
    }
    out << ']';
    out << ",\"postprocess\":{";
    out << "\"thresh\":";
    append_float(out, config_.postprocess.thresh);
    out << ",\"box_thresh\":";
    append_float(out, config_.postprocess.box_thresh);
    out << ",\"unclip_ratio\":";
    append_float(out, config_.postprocess.unclip_ratio);
    out << ",\"max_candidates\":" << config_.postprocess.max_candidates;
    out << ",\"min_size\":" << config_.postprocess.min_size;
    out << '}';
    out << ",\"detector_preprocess\":{";
    out << "\"limit_side_len\":" << config_.detector_preprocess.limit_side_len;
    out << ",\"limit_type\":";
    append_json_escaped(out,
                        limit_type_to_string(config_.detector_preprocess.limit_type));
    out << ",\"max_side_limit\":"
        << config_.detector_preprocess.max_side_limit;
#if defined(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
    out << ",\"implementation\":\"opencv\"";
#else
    out << ",\"implementation\":\"fallback_smoke\"";
#endif
    out << '}';
    out << '}';
    return out.str();
  }

  [[nodiscard]] const FullPageWorkerConfig &config() const noexcept {
    return config_;
  }

private:
  void copy_source_image(const std::uint8_t *image, int image_height,
                         int /*image_width*/, int image_stride) {
    const std::size_t bytes =
        static_cast<std::size_t>(image_height) *
        static_cast<std::size_t>(image_stride);
    if (source_capacity_bytes_ < bytes) {
      d_source_image_.reset(bytes);
      source_capacity_bytes_ = bytes;
    }
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(d_source_image_.get(), image, bytes,
                                       cudaMemcpyHostToDevice, stream_));
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  FullPageWorkerConfig config_;
  detection::DetectionWorker detector_;
  recognition::RecognitionWorker recognizer_;
  int max_bucket_width_ = 0;
  CudaPtr<std::uint8_t> d_source_image_;
  CudaPtr<float> d_m_invs_;
  CudaPtr<int> d_crop_widths_;
  CudaPtr<float> d_roi_batch_;
  std::size_t source_capacity_bytes_ = 0;
  cudaStream_t stream_ = nullptr;
  std::mutex mutex_;
};

FullPageWorker::FullPageWorker(FullPageWorkerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

FullPageWorker::~FullPageWorker() = default;

FullPageResult FullPageWorker::recognize_page(
    const std::uint8_t *image, int image_height, int image_width,
    int image_stride, SourceColorOrder source_color_order,
    const float *detector_nchw, int detector_height, int detector_width) {
  return impl_->recognize_page(image, image_height, image_width, image_stride,
                               source_color_order, detector_nchw,
                               detector_height, detector_width);
}

FullPageResult FullPageWorker::recognize_page_image(
    const std::uint8_t *image, int image_height, int image_width,
    int image_stride, SourceColorOrder source_color_order) {
  return impl_->recognize_page_image(image, image_height, image_width,
                                     image_stride, source_color_order);
}

std::string FullPageWorker::info_json() const { return impl_->info_json(); }

const FullPageWorkerConfig &FullPageWorker::config() const noexcept {
  return impl_->config();
}

std::string full_page_result_to_json(const FullPageResult &result) {
  const double lines_per_second =
      result.timing.total_ms > 0.0
          ? static_cast<double>(result.lines.size()) /
                (result.timing.total_ms / 1000.0)
          : 0.0;

  std::ostringstream out;
  out << '{';
  out << "\"count\":" << result.lines.size();
  out << ",\"source_shape\":[" << result.source_height << ','
      << result.source_width << ']';
  out << ",\"detector_input_shape\":[1,3," << result.detector_height << ','
      << result.detector_width << ']';
  out << ",\"detector_output_shape\":[1,1,"
      << result.detector_output_height << ',' << result.detector_output_width
      << ']';
  out << ",\"lines_per_second\":";
  append_float(out, lines_per_second);
  out << ",\"timing\":{";
  out << "\"total_ms\":";
  append_float(out, result.timing.total_ms);
  out << ",\"detector_preprocess_ms\":";
  append_float(out, result.timing.detector_preprocess_ms);
  out << ",\"detect_ms\":";
  append_float(out, result.timing.detect_ms);
  out << ",\"postprocess_ms\":";
  append_float(out, result.timing.postprocess_ms);
  out << ",\"roi_ms\":";
  append_float(out, result.timing.roi_ms);
  out << ",\"recognize_ms\":";
  append_float(out, result.timing.recognize_ms);
  out << ",\"detector_h2d_ms\":";
  append_float(out, result.timing.detector_h2d_ms);
  out << ",\"detector_execute_ms\":";
  append_float(out, result.timing.detector_execute_ms);
  out << ",\"detector_d2h_ms\":";
  append_float(out, result.timing.detector_d2h_ms);
  out << ",\"detector_output_bytes\":"
      << result.timing.detector_output_bytes;
  out << ",\"recognition_output_bytes\":"
      << result.timing.recognition_output_bytes;
  out << '}';

  out << ",\"recognition_chunks\":[";
  for (std::size_t i = 0; i < result.recognition_chunks.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto &chunk = result.recognition_chunks[i];
    out << "{\"batch\":" << chunk.batch << ",\"width\":" << chunk.width
        << ",\"timesteps\":" << chunk.timesteps << '}';
  }
  out << ']';

  out << ",\"lines\":[";
  for (std::size_t i = 0; i < result.lines.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto &line = result.lines[i];
    out << '{';
    out << "\"index\":" << line.index;
    out << ",\"text\":";
    append_json_escaped(out, line.prediction.decoded.text);
    out << ",\"score\":";
    append_float(out, line.prediction.decoded.score);
    out << ",\"det_score\":";
    append_float(out, line.detection.score);
    out << ",\"bucket_width\":" << line.bucket_width;
    out << ",\"crop_width\":" << line.crop_width;
    out << ",\"natural_width\":" << line.natural_width;
    out << ",\"vertical\":" << (line.vertical ? "true" : "false");
    out << ",\"clipped\":" << (line.clipped ? "true" : "false");
    out << ",\"component_pixels\":" << line.detection.component_pixels;
    out << ",\"class_ids\":";
    append_int_array(out, line.prediction.decoded.class_ids);
    out << ",\"per_char_scores\":";
    append_float_array(out, line.prediction.decoded.per_char_scores);
    out << ",\"box\":[";
    for (std::size_t point_index = 0;
         point_index < line.detection.box.points.size(); ++point_index) {
      if (point_index != 0) {
        out << ',';
      }
      const auto &point = line.detection.box.points[point_index];
      out << '[';
      append_float(out, point.x);
      out << ',';
      append_float(out, point.y);
      out << ']';
    }
    out << "]}";
  }
  out << ']';
  out << '}';
  return out.str();
}

} // namespace ppocrv6_native::full_page
