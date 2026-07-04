#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ppocrv6_native::engine {

struct RecognitionProfile {
  int min_batch = 1;
  int opt_batch = 8;
  int max_batch = 12;
  int channels = 3;
  int height = 48;
  int min_width = 48;
  int opt_width = 640;
  int max_width = 3200;
  std::size_t workspace_bytes = 1024ULL * 1024ULL * 1024ULL;
  int builder_optimization_level = 3;
  bool fp16 = true;
};

struct DetectionProfile {
  int min_batch = 1;
  int opt_batch = 1;
  int max_batch = 1;
  int channels = 3;
  int min_height = 32;
  int opt_height = 736;
  int max_height = 1280;
  int min_width = 32;
  int opt_width = 736;
  int max_width = 1280;
  std::size_t workspace_bytes = 1024ULL * 1024ULL * 1024ULL;
  int builder_optimization_level = 3;
  bool fp16 = false;
};

bool build_recognition_hidden_engine(const std::string &onnx_path,
                                     const std::string &engine_path,
                                     const RecognitionProfile &profile);

bool build_recognition_hidden_engine(const std::string &onnx_path,
                                     const std::string &engine_path,
                                     const std::vector<RecognitionProfile> &profiles);

bool build_detection_engine(const std::string &onnx_path,
                            const std::string &engine_path,
                            const DetectionProfile &profile);

} // namespace ppocrv6_native::engine
