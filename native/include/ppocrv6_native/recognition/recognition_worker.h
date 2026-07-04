#pragma once

#include "ppocrv6_native/recognition/ctc_decode.h"

#include <memory>
#include <string>
#include <vector>

namespace ppocrv6_native::recognition {

struct RecognitionWorkerConfig {
  std::string engine_path = "artifacts/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt";
  std::string weight_path = "artifacts/ppocrv6-medium/classifier/weight.fp16.bin";
  std::string bias_path = "artifacts/ppocrv6-medium/classifier/bias.fp16.bin";
  std::string characters_path = "artifacts/ppocrv6-medium/classifier/characters.txt";
  int default_width = 80;
  int default_batch_size = 64;
  int max_batch_size = 256;
  int max_width = 128;
  int vocab_size = 18710;
  int hidden_size = 192;
  int vocab_tile_size = 512;
  int blank_id = 0;
  int warmup_runs = 1;
};

struct RecognitionPrediction {
  DecodedText decoded;
  std::vector<int> timestep_class_ids;
  std::vector<float> timestep_scores;
};

struct RecognitionChunkMeta {
  int batch = 0;
  int width = 0;
  int timesteps = 0;
};

struct RecognitionBatchResult {
  int count = 0;
  int width = 0;
  int batch_size = 0;
  double elapsed_ms = 0.0;
  std::size_t output_bytes = 0;
  std::vector<RecognitionChunkMeta> chunks;
  std::vector<RecognitionPrediction> predictions;
};

class RecognitionWorker {
public:
  explicit RecognitionWorker(RecognitionWorkerConfig config);
  ~RecognitionWorker();

  RecognitionWorker(const RecognitionWorker &) = delete;
  RecognitionWorker &operator=(const RecognitionWorker &) = delete;

  RecognitionBatchResult recognize_f32(const float *nchw, int count, int width,
                                       int batch_size,
                                       bool return_timesteps);
  RecognitionBatchResult recognize_device_f32(const float *device_nchw,
                                              int count, int width,
                                              int batch_size,
                                              bool return_timesteps);

  [[nodiscard]] std::string info_json() const;
  [[nodiscard]] const RecognitionWorkerConfig &config() const noexcept;
  [[nodiscard]] int classes() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::string recognition_result_to_json(const RecognitionBatchResult &result,
                                       bool include_timesteps);

} // namespace ppocrv6_native::recognition
