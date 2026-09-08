#pragma once

#include "ppocrv6_native/detection/db_postprocess.h"
#include "ppocrv6_native/detection/detection_worker.h"
#include "ppocrv6_native/recognition/recognition_worker.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ppocrv6_native::full_page {

enum class SourceColorOrder {
  kRgb = 0,
  kBgr = 1,
};

enum class DetectorLimitType {
  kMax = 0,
  kMin = 1,
  kResizeLong = 2,
};

struct DetectorPreprocessConfig {
  int limit_side_len = 1280;
  DetectorLimitType limit_type = DetectorLimitType::kMax;
  int max_side_limit = 4000;
};

struct FullPageWorkerConfig {
  detection::DetectionWorkerConfig detector{};
  DetectorPreprocessConfig detector_preprocess{};
  recognition::RecognitionWorkerConfig recognizer = [] {
    recognition::RecognitionWorkerConfig config;
    config.default_width = 1600;
    config.default_batch_size = 8;
    config.max_batch_size = 12;
    config.max_width = 3200;
    return config;
  }();
  detection::DetectionPostprocessConfig postprocess{};
  std::vector<int> recognition_buckets{
      128, 256, 384, 512, 640, 960, 1280, 1600, 2400, 3200};
  int recognition_height = 48;
};

struct FullPageLine {
  int index = 0;
  detection::DetectionBox detection;
  int bucket_width = 0;
  int crop_width = 0;
  int natural_width = 0;
  bool vertical = false;
  bool clipped = false;
  recognition::RecognitionPrediction prediction;
};

struct FullPageTiming {
  double total_ms = 0.0;
  double detector_preprocess_ms = 0.0;
  double detect_ms = 0.0;
  double postprocess_ms = 0.0;
  double roi_ms = 0.0;
  double recognize_ms = 0.0;
  float detector_h2d_ms = 0.0f;
  float detector_execute_ms = 0.0f;
  float detector_d2h_ms = 0.0f;
  std::size_t detector_output_bytes = 0;
  std::size_t recognition_output_bytes = 0;
};

struct FullPageResult {
  int source_height = 0;
  int source_width = 0;
  int detector_height = 0;
  int detector_width = 0;
  int detector_output_height = 0;
  int detector_output_width = 0;
  FullPageTiming timing;
  std::vector<recognition::RecognitionChunkMeta> recognition_chunks;
  std::vector<FullPageLine> lines;
};

class FullPageWorker {
public:
  explicit FullPageWorker(FullPageWorkerConfig config,
      std::shared_ptr<recognition::RecognitionWorker> recognizer = nullptr);
  ~FullPageWorker();

  FullPageWorker(const FullPageWorker &) = delete;
  FullPageWorker &operator=(const FullPageWorker &) = delete;

  FullPageResult recognize_page(const std::uint8_t *image, int image_height,
                                int image_width, int image_stride,
                                SourceColorOrder source_color_order,
                                const float *detector_nchw,
                                int detector_height, int detector_width);

  FullPageResult recognize_page_image(const std::uint8_t *image,
                                      int image_height, int image_width,
                                      int image_stride,
                                      SourceColorOrder source_color_order);

  [[nodiscard]] std::string info_json() const;
  [[nodiscard]] const FullPageWorkerConfig &config() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::string full_page_result_to_json(const FullPageResult &result);

} // namespace ppocrv6_native::full_page
