#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ppocrv6_native::detection {

struct DetectionWorkerConfig {
  std::string engine_path = "artifacts/ppocrv6-medium/engines/det-fp16-b1-h256-1280-w256-1280.trt";
  int default_batch = 1;
  int default_height = 1280;
  int default_width = 992;
  int max_batch = 1;
  int max_height = 1280;
  int max_width = 1280;
  int warmup_runs = 1;
};

struct DetectionTensorResult {
  int batch = 0;
  int height = 0;
  int width = 0;
  int output_n = 0;
  int output_c = 0;
  int output_h = 0;
  int output_w = 0;
  double elapsed_ms = 0.0;
  float h2d_ms = 0.0f;
  float execute_ms = 0.0f;
  float d2h_ms = 0.0f;
  std::size_t output_bytes = 0;
  std::vector<float> output;
};

class DetectionWorker {
public:
  explicit DetectionWorker(DetectionWorkerConfig config);
  ~DetectionWorker();

  DetectionWorker(const DetectionWorker &) = delete;
  DetectionWorker &operator=(const DetectionWorker &) = delete;

  DetectionTensorResult detect_f32(const float *nchw, int batch, int height,
                                   int width, bool copy_output);

  [[nodiscard]] std::string info_json() const;
  [[nodiscard]] const DetectionWorkerConfig &config() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::string detection_result_to_json(const DetectionTensorResult &result);

} // namespace ppocrv6_native::detection
