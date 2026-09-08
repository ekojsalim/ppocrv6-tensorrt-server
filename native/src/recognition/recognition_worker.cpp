#include "ppocrv6_native/recognition/recognition_worker.h"

#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/common/trt_workspace.h"
#include "ppocrv6_native/engine/trt_engine.h"
#include "ppocrv6_native/kernels/classifier.h"
#include "ppocrv6_native/recognition/glyph_shape.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ppocrv6_native::recognition {

const char *character_policy_name(CharacterPolicy policy) noexcept {
  switch (policy) {
  case CharacterPolicy::kCjkFocusFallback:
    return "cjk_focus_fallback";
  case CharacterPolicy::kCjkFocus:
    return "cjk_focus";
  case CharacterPolicy::kSuppressAscii:
    return "suppress_ascii";
  case CharacterPolicy::kAll:
  default:
    return "all";
  }
}

const char *score_mode_name(ScoreMode mode) noexcept {
  return mode == ScoreMode::kAccepted ? "accepted" : "model";
}

namespace {

constexpr float kOneStrokeFallbackMinProbability = 0.005f;
constexpr const char *kOneToken = "\xE4\xB8\x80";

class Logger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[ppocrv6-recognizer] " << msg << '\n';
    }
  }
};

std::vector<char> read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  const std::streamsize bytes = in.tellg();
  if (bytes <= 0) {
    throw std::runtime_error("invalid file size for " + path);
  }
  std::vector<char> out(static_cast<std::size_t>(bytes));
  in.seekg(0, std::ios::beg);
  in.read(out.data(), bytes);
  if (!in) {
    throw std::runtime_error("failed to read " + path);
  }
  return out;
}

template <typename T>
std::vector<T> read_binary(const std::string &path, std::size_t expected_count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  const std::streamsize bytes = in.tellg();
  const std::size_t expected_bytes = expected_count * sizeof(T);
  if (bytes < 0 || static_cast<std::size_t>(bytes) != expected_bytes) {
    throw std::runtime_error("unexpected size for " + path + ": got " +
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

std::vector<std::string> read_characters(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open characters file " + path);
  }
  std::vector<std::string> characters;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    characters.push_back(line);
  }
  if (characters.empty()) {
    throw std::runtime_error("characters file is empty: " + path);
  }
  return characters;
}

bool is_ascii_token(const std::string &token) {
  return !token.empty() &&
         std::all_of(token.begin(), token.end(), [](unsigned char byte) {
           return byte < 0x80;
         });
}

bool is_cjk_focus_extra_suppressed_token(const std::string &token) {
  return token == "\xC2\xAF" ||     // MACRON
         token == "\xE2\x80\x93" || // EN DASH
         token == "\xE2\x80\x94" || // EM DASH
         token == "\xE2\x80\x95" || // HORIZONTAL BAR
         token == "\xE2\x88\x92" || // MINUS SIGN
         token == "\xE2\x8E\xAF" || // HORIZONTAL LINE EXTENSION
         token == "\xEF\xBC\x8D";   // FULLWIDTH HYPHEN-MINUS
}

std::size_t volume(const nvinfer1::Dims &dims) {
  if (dims.nbDims <= 0) {
    return 0;
  }
  std::size_t out = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      return 0;
    }
    out *= static_cast<std::size_t>(dims.d[i]);
  }
  return out;
}

bool dims_contains(const nvinfer1::Dims &min_dims,
                   const nvinfer1::Dims &max_dims,
                   const nvinfer1::Dims &dims) {
  if (min_dims.nbDims != dims.nbDims || max_dims.nbDims != dims.nbDims) {
    return false;
  }
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < min_dims.d[i] || dims.d[i] > max_dims.d[i]) {
      return false;
    }
  }
  return true;
}

std::string discover_input_name(nvinfer1::ICudaEngine &engine) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char *name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      return name;
    }
  }
  throw std::runtime_error("failed to discover TensorRT input name");
}

std::string discover_output_name(nvinfer1::ICudaEngine &engine) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char *name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      return name;
    }
  }
  throw std::runtime_error("failed to discover TensorRT output name");
}

struct SharedResources {
  explicit SharedResources(const RecognitionWorkerConfig &config) {
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
      throw std::runtime_error("failed to create TensorRT runtime");
    }
    const auto engine_bytes = read_file(config.engine_path);
    engine.reset(
        runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!engine) {
      throw std::runtime_error("failed to deserialize engine " +
                               config.engine_path);
    }
    input_name = discover_input_name(*engine);
    output_name = discover_output_name(*engine);

    const std::size_t weight_count =
        static_cast<std::size_t>(config.hidden_size) *
        static_cast<std::size_t>(config.vocab_size);
    const std::size_t bias_count = static_cast<std::size_t>(config.vocab_size);
    const auto host_weight =
        read_binary<std::uint16_t>(config.weight_path, weight_count);
    const auto host_bias =
        read_binary<std::uint16_t>(config.bias_path, bias_count);
    characters = read_characters(config.characters_path);
    if (characters.size() != bias_count) {
      throw std::runtime_error(
          "characters count does not match classifier vocabulary");
    }
    auto host_ascii_suppressed_bias = host_bias;
    auto host_cjk_focus_bias = host_bias;
    constexpr std::uint16_t kHalfNegativeInfinity = 0xfc00U;
    for (std::size_t class_id = 0; class_id < characters.size(); ++class_id) {
      if (characters[class_id] == kOneToken) {
        one_class_id = static_cast<int>(class_id);
      }
      if (static_cast<int>(class_id) == config.blank_id) {
        continue;
      }
      if (is_ascii_token(characters[class_id])) {
        host_ascii_suppressed_bias[class_id] = kHalfNegativeInfinity;
        host_cjk_focus_bias[class_id] = kHalfNegativeInfinity;
        ++ascii_suppressed_class_count;
        ++cjk_focus_suppressed_class_count;
      } else if (is_cjk_focus_extra_suppressed_token(characters[class_id])) {
        host_cjk_focus_bias[class_id] = kHalfNegativeInfinity;
        ++cjk_focus_extra_suppressed_class_count;
        ++cjk_focus_suppressed_class_count;
      }
    }
    if (one_class_id < 0) {
      throw std::runtime_error("characters file does not contain 一");
    }

    weight.reset(weight_count);
    bias.reset(bias_count);
    ascii_suppressed_bias.reset(bias_count);
    cjk_focus_bias.reset(bias_count);
    PPOCRV6_CUDA_CHECK(cudaMemcpy(weight.get(), host_weight.data(),
                                  host_weight.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice));
    PPOCRV6_CUDA_CHECK(cudaMemcpy(bias.get(), host_bias.data(),
                                  host_bias.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice));
    PPOCRV6_CUDA_CHECK(cudaMemcpy(
        ascii_suppressed_bias.get(), host_ascii_suppressed_bias.data(),
        host_ascii_suppressed_bias.size() * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice));
    PPOCRV6_CUDA_CHECK(cudaMemcpy(
        cjk_focus_bias.get(), host_cjk_focus_bias.data(),
        host_cjk_focus_bias.size() * sizeof(std::uint16_t),
        cudaMemcpyHostToDevice));
  }

  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  CudaPtr<std::uint16_t> weight;
  CudaPtr<std::uint16_t> bias;
  CudaPtr<std::uint16_t> ascii_suppressed_bias;
  CudaPtr<std::uint16_t> cjk_focus_bias;
  std::size_t ascii_suppressed_class_count = 0;
  std::size_t cjk_focus_extra_suppressed_class_count = 0;
  std::size_t cjk_focus_suppressed_class_count = 0;
  int one_class_id = -1;
  std::vector<std::string> characters;
  std::string input_name;
  std::string output_name;
};

std::string resource_cache_key(const RecognitionWorkerConfig &config) {
  std::ostringstream out;
  out << config.engine_path << '\0' << config.weight_path << '\0'
      << config.bias_path << '\0' << config.characters_path << '\0'
      << config.vocab_size << ':' << config.hidden_size << ':'
      << config.blank_id;
  return out.str();
}

std::shared_ptr<SharedResources>
acquire_shared_resources(const RecognitionWorkerConfig &config) {
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, std::weak_ptr<SharedResources>> cache;

  const auto key = resource_cache_key(config);
  std::lock_guard<std::mutex> guard(cache_mutex);
  if (const auto found = cache.find(key); found != cache.end()) {
    if (auto shared = found->second.lock()) {
      return shared;
    }
  }
  auto shared = std::make_shared<SharedResources>(config);
  cache[key] = shared;
  return shared;
}

int find_profile(nvinfer1::ICudaEngine &engine, const std::string &input_name,
                 int batch, int width) {
  const nvinfer1::Dims4 dims{batch, 3, 48, width};
  const int profiles = engine.getNbOptimizationProfiles();
  int best_profile = -1;
  int64_t best_memory = std::numeric_limits<int64_t>::max();
  for (int profile = 0; profile < profiles; ++profile) {
    const auto min_dims = engine.getProfileShape(
        input_name.c_str(), profile, nvinfer1::OptProfileSelector::kMIN);
    const auto max_dims = engine.getProfileShape(
        input_name.c_str(), profile, nvinfer1::OptProfileSelector::kMAX);
    if (dims_contains(min_dims, max_dims, dims)) {
      const int64_t memory = engine.getDeviceMemorySizeForProfileV2(profile);
      if (memory < best_memory) {
        best_profile = profile;
        best_memory = memory;
      }
    }
  }
  return best_profile;
}

int resolve_profile(nvinfer1::ICudaEngine &engine,
                    const std::string &input_name, int batch, int width) {
  const int profile = find_profile(engine, input_name, batch, width);
  if (profile >= 0) {
    return profile;
  }
  const nvinfer1::Dims4 dims{batch, 3, 48, width};
  throw std::runtime_error("no TensorRT optimization profile accepts shape " +
                           ppocrv6_native::engine::dims_to_string(dims));
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

void append_float(std::ostringstream &out, float value) {
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

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

} // namespace

class RecognitionWorker::Impl {
public:
  explicit Impl(RecognitionWorkerConfig config) : config_(std::move(config)) {
    if (config_.max_batch_size <= 0 || config_.max_width <= 0 ||
        config_.default_batch_size <= 0 || config_.default_width <= 0 ||
        config_.vocab_size <= 0 || config_.hidden_size <= 0 ||
        config_.vocab_tile_size <= 0) {
      throw std::runtime_error("invalid recognition worker config");
    }
    if (config_.default_batch_size > config_.max_batch_size ||
        config_.default_width > config_.max_width) {
      throw std::runtime_error(
          "default recognition shape must not exceed max shape");
    }

    PPOCRV6_CUDA_CHECK(cudaFree(nullptr));
    resources_ = acquire_shared_resources(config_);
    workspace_ = acquire_trt_workspace(static_cast<std::size_t>(
        std::max<int64_t>(0, resources_->engine->getDeviceMemorySizeV2())));
    context_.reset(resources_->engine->createExecutionContext(
        nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!context_) {
      throw std::runtime_error("failed to create TensorRT execution context");
    }

    PPOCRV6_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    allocate_for_max_shape();
    warmup();
  }

  ~Impl() {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  RecognitionBatchResult recognize_f32(const float *nchw, int count, int width,
                                       int batch_size,
                                       bool return_timesteps,
                                       CharacterPolicy character_policy,
                                       ScoreMode score_mode) {
    if (nchw == nullptr) {
      throw std::runtime_error("input tensor pointer is null");
    }
    if (count <= 0) {
      throw std::runtime_error("count must be positive");
    }
    if (width <= 0 || width > config_.max_width) {
      throw std::runtime_error("width is outside worker limits");
    }
    if (batch_size <= 0 || batch_size > config_.max_batch_size) {
      throw std::runtime_error("batch_size is outside worker limits");
    }

    std::lock_guard<std::mutex> guard(mutex_);
    RecognitionBatchResult result;
    result.count = count;
    result.width = width;
    result.batch_size = batch_size;
    result.character_policy = character_policy;
    result.score_mode = score_mode;
    result.predictions.reserve(static_cast<std::size_t>(count));
    const auto started = std::chrono::steady_clock::now();

    const std::size_t image_stride =
        3U * 48U * static_cast<std::size_t>(width);
    for (int start = 0; start < count; start += batch_size) {
      const int chunk_count = std::min(batch_size, count - start);
      run_chunk(nchw + static_cast<std::size_t>(start) * image_stride,
                chunk_count, width, return_timesteps, character_policy,
                score_mode, result);
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));

    result.elapsed_ms =
        elapsed_ms(started, std::chrono::steady_clock::now());
    return result;
  }

  RecognitionBatchResult recognize_device_f32(const float *device_nchw,
                                              int count, int width,
                                              int batch_size,
                                              bool return_timesteps,
                                              CharacterPolicy character_policy,
                                              ScoreMode score_mode) {
    if (device_nchw == nullptr) {
      throw std::runtime_error("device input tensor pointer is null");
    }
    if (count <= 0) {
      throw std::runtime_error("count must be positive");
    }
    if (width <= 0 || width > config_.max_width) {
      throw std::runtime_error("width is outside worker limits");
    }
    if (batch_size <= 0 || batch_size > config_.max_batch_size) {
      throw std::runtime_error("batch_size is outside worker limits");
    }

    return recognize_device_f32_on_stream(device_nchw, count, width, batch_size,
                                          return_timesteps, stream_,
                                          character_policy, score_mode);
  }

  RecognitionBatchResult recognize_device_f32_on_stream(
      const float *device_nchw, int count, int width, int batch_size,
      bool return_timesteps, cudaStream_t stream,
      CharacterPolicy character_policy, ScoreMode score_mode) {
    if (stream == nullptr) {
      throw std::runtime_error("recognition CUDA stream is null");
    }
    if (device_nchw == nullptr) {
      throw std::runtime_error("device input tensor pointer is null");
    }
    if (count <= 0) {
      throw std::runtime_error("count must be positive");
    }
    if (width <= 0 || width > config_.max_width) {
      throw std::runtime_error("width is outside worker limits");
    }
    if (batch_size <= 0 || batch_size > config_.max_batch_size) {
      throw std::runtime_error("batch_size is outside worker limits");
    }

    std::lock_guard<std::mutex> guard(mutex_);
    RecognitionBatchResult result;
    result.count = count;
    result.width = width;
    result.batch_size = batch_size;
    result.character_policy = character_policy;
    result.score_mode = score_mode;
    result.predictions.reserve(static_cast<std::size_t>(count));
    const auto started = std::chrono::steady_clock::now();

    const std::size_t image_stride =
        3U * 48U * static_cast<std::size_t>(width);
    for (int start = 0; start < count; start += batch_size) {
      const int chunk_count = std::min(batch_size, count - start);
      run_chunk_device(device_nchw +
                           static_cast<std::size_t>(start) * image_stride,
                       chunk_count, width, return_timesteps, character_policy,
                       score_mode, result, stream);
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));

    result.elapsed_ms =
        elapsed_ms(started, std::chrono::steady_clock::now());
    return result;
  }

  [[nodiscard]] std::string info_json() const {
    std::lock_guard<std::mutex> guard(mutex_);
    std::ostringstream out;
    out << '{';
    out << "\"engine\":";
    append_json_escaped(out, config_.engine_path);
    out << ",\"weight\":";
    append_json_escaped(out, config_.weight_path);
    out << ",\"bias\":";
    append_json_escaped(out, config_.bias_path);
    out << ",\"characters\":";
    append_json_escaped(out, config_.characters_path);
    out << ",\"providers\":[\"TensorRT\",\"CUDA\"]";
    out << ",\"outputs\":[\"ids\",\"prob\"]";
    out << ",\"classes\":" << resources_->characters.size();
    out << ",\"default_width\":" << config_.default_width;
    out << ",\"default_batch_size\":" << config_.default_batch_size;
    out << ",\"max_width\":" << config_.max_width;
    out << ",\"max_batch_size\":" << config_.max_batch_size;
    out << ",\"vocab_size\":" << config_.vocab_size;
    out << ",\"hidden_size\":" << config_.hidden_size;
    out << ",\"vocab_tile_size\":" << config_.vocab_tile_size;
    out << ",\"supported_score_modes\":[\"accepted\",\"model\"]";
    out << ",\"accepted_score_type\":\"binary_acceptance\"";
    out << ",\"model_score_type\":\"probability_or_conditional_probability\"";
    out << ",\"accepted_probability_calculation\":\"skipped\"";
    out << ",\"supported_character_policies\":[\"cjk_focus\","
           "\"cjk_focus_fallback\",\"suppress_ascii\",\"all\"]";
    out << ",\"ascii_suppressed_class_count\":"
        << resources_->ascii_suppressed_class_count;
    out << ",\"cjk_focus_extra_suppressed_class_count\":"
        << resources_->cjk_focus_extra_suppressed_class_count;
    out << ",\"cjk_focus_suppressed_class_count\":"
        << resources_->cjk_focus_suppressed_class_count;
    out << ",\"one_stroke_fallback_token\":\"" << kOneToken << '"';
    out << ",\"one_stroke_fallback_class_id\":"
        << resources_->one_class_id;
    out << ",\"one_stroke_fallback_min_probability\":"
        << kOneStrokeFallbackMinProbability;
    out << ",\"context_mode\":\"switch\"";
    out << ",\"profiles\":" << resources_->engine->getNbOptimizationProfiles();
    out << ",\"shared_engine_use_count\":" << resources_.use_count();
    out << ",\"context_memory_bytes\":" << context_memory_capacity_bytes_;
    out << ",\"workspace_shared\":" << (shared_trt_workspace_enabled() ? "true" : "false");
    out << ",\"workspace_reserved_bytes\":" << workspace_->size();
    out << ",\"profile_memory_bytes\":[";
    for (int profile = 0;
         profile < resources_->engine->getNbOptimizationProfiles(); ++profile) {
      if (profile != 0) {
        out << ',';
      }
      out << std::max<int64_t>(
          0, resources_->engine->getDeviceMemorySizeForProfileV2(profile));
    }
    out << ']';
    out << '}';
    return out.str();
  }

  [[nodiscard]] const RecognitionWorkerConfig &config() const noexcept {
    return config_;
  }

  [[nodiscard]] int classes() const noexcept {
    return static_cast<int>(resources_->characters.size());
  }

  [[nodiscard]] bool supports_shape(int batch, int width) const {
    if (batch <= 0 || width <= 0 || batch > config_.max_batch_size ||
        width > config_.max_width) {
      return false;
    }
    return find_profile(*resources_->engine, resources_->input_name, batch,
                        width) >= 0;
  }

private:
  void allocate_for_max_shape() {
    TrtWorkspace::Lease lease(*workspace_, stream_);
    set_active_shape(config_.max_batch_size, config_.max_width, stream_, false);
    max_timesteps_ = output_dims_.d[1];
    max_rows_ = config_.max_batch_size * max_timesteps_;
    max_input_count_ =
        static_cast<std::size_t>(config_.max_batch_size) * 3U * 48U *
        static_cast<std::size_t>(config_.max_width);
    max_hidden_count_ = volume(output_dims_);
    const auto workspace = kernels::tiled_classifier_workspace_shape(
        max_rows_, config_.vocab_size, 16, config_.vocab_tile_size);


    hidden_.reset(max_hidden_count_);
    indices_.reset(static_cast<std::size_t>(max_rows_));
    prob_.reset(static_cast<std::size_t>(max_rows_));
    max_logits_.reset(static_cast<std::size_t>(max_rows_));
    one_vs_blank_prob_.reset(static_cast<std::size_t>(max_rows_));
    partial_ids_.reset(workspace.partial_count);
    partial_max_.reset(workspace.partial_count);
    partial_sum_.reset(workspace.partial_count);
    host_indices_.reset(static_cast<std::size_t>(max_rows_));
    host_prob_.reset(static_cast<std::size_t>(max_rows_));
    host_one_vs_blank_prob_.reset(static_cast<std::size_t>(max_rows_));
  }

  void warmup() {
    if (config_.warmup_runs <= 0) {
      return;
    }
    TrtWorkspace::Lease lease(*workspace_, stream_);
    input_.reset(max_input_count_);
    const int warmup_batch =
        std::min(config_.default_batch_size, config_.max_batch_size);
    set_active_shape(warmup_batch, config_.default_width, stream_);
    const std::size_t input_count =
        static_cast<std::size_t>(warmup_batch) * 3U * 48U *
        static_cast<std::size_t>(config_.default_width);
    PPOCRV6_CUDA_CHECK(
        cudaMemsetAsync(input_.get(), 0, input_count * sizeof(float), stream_));
    for (int i = 0; i < config_.warmup_runs; ++i) {
      launch_compute(input_.get(), stream_, CharacterPolicy::kAll, true, lease);
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  void ensure_context_memory(int profile, cudaStream_t stream) {
    (void)stream;
    context_memory_capacity_bytes_ = static_cast<std::size_t>(std::max<int64_t>(
        0, resources_->engine->getDeviceMemorySizeForProfileV2(profile)));
    // The caller holds a workspace lease. Bind in launch_compute after shape
    // selection; another worker may have grown the arena since our last call.
  }

  void set_active_shape(int batch, int width, cudaStream_t stream,
                        bool allocate_context_memory = true) {
    const int profile = resolve_profile(*resources_->engine,
                                        resources_->input_name, batch, width);
    if (profile != current_profile_) {
      if (!context_->setOptimizationProfileAsync(profile, stream)) {
        throw std::runtime_error("failed to set TensorRT optimization profile " +
                                 std::to_string(profile));
      }
      current_profile_ = profile;
    }
    if (allocate_context_memory) {
      ensure_context_memory(profile, stream);
    }
    input_dims_ = nvinfer1::Dims4{batch, 3, 48, width};
    if (!context_->setInputShape(resources_->input_name.c_str(), input_dims_)) {
      throw std::runtime_error("failed to set TensorRT input shape " +
                               ppocrv6_native::engine::dims_to_string(
                                   input_dims_));
    }
    output_dims_ = context_->getTensorShape(resources_->output_name.c_str());
    if (output_dims_.nbDims != 3 || output_dims_.d[0] != batch ||
        output_dims_.d[2] != config_.hidden_size) {
      throw std::runtime_error("unexpected hidden shape " +
                               ppocrv6_native::engine::dims_to_string(
                                   output_dims_));
    }
    active_batch_ = batch;
    active_width_ = width;
    active_timesteps_ = output_dims_.d[1];
    active_rows_ = active_batch_ * active_timesteps_;
    const std::size_t input_count =
        static_cast<std::size_t>(active_batch_) * 3U * 48U *
        static_cast<std::size_t>(active_width_);
    const std::size_t hidden_count = volume(output_dims_);
    if ((max_input_count_ != 0 && input_count > max_input_count_) ||
        (max_hidden_count_ != 0 && hidden_count > max_hidden_count_) ||
        (max_rows_ != 0 && active_rows_ > max_rows_)) {
      throw std::runtime_error("active shape exceeds worker buffers");
    }
  }

  const half *classifier_bias_for_policy(
      CharacterPolicy character_policy) const {
    switch (character_policy) {
    case CharacterPolicy::kCjkFocus:
    case CharacterPolicy::kCjkFocusFallback:
      return reinterpret_cast<const half *>(
          resources_->cjk_focus_bias.get());
    case CharacterPolicy::kSuppressAscii:
      return reinterpret_cast<const half *>(
          resources_->ascii_suppressed_bias.get());
    case CharacterPolicy::kAll:
    default:
      return reinterpret_cast<const half *>(resources_->bias.get());
    }
  }

  void launch_classifier(const half *classifier_bias, cudaStream_t stream,
                         bool calculate_probability) {
    kernels::cuda_linear_argmax_prob_wmma_mode(
        reinterpret_cast<const half *>(hidden_.get()),
        reinterpret_cast<const half *>(resources_->weight.get()),
        classifier_bias, indices_.get(), prob_.get(), max_logits_.get(),
        active_rows_, config_.hidden_size, config_.vocab_size,
        config_.vocab_tile_size, partial_ids_.get(), partial_max_.get(),
        partial_sum_.get(), calculate_probability, stream);
  }

  void launch_compute(const float *device_input, cudaStream_t stream,
                      CharacterPolicy character_policy,
                      bool calculate_probability, TrtWorkspace::Lease &lease) {
    context_->setDeviceMemoryV2(lease.data(), static_cast<int64_t>(lease.size()));
    // All callers hold a lease through stream completion.
    if (!context_->setTensorAddress(resources_->input_name.c_str(),
                                    const_cast<float *>(device_input)) ||
        !context_->setTensorAddress(resources_->output_name.c_str(),
                                    hidden_.get())) {
      throw std::runtime_error("failed to set TensorRT tensor addresses");
    }
    if (!context_->enqueueV3(stream)) {
      throw std::runtime_error("TensorRT hidden enqueue failed");
    }
    launch_classifier(classifier_bias_for_policy(character_policy), stream,
                      calculate_probability);
  }

  void run_chunk(const float *chunk_input, int chunk_count, int width,
                 bool return_timesteps, CharacterPolicy character_policy,
                 ScoreMode score_mode, RecognitionBatchResult &result) {
    TrtWorkspace::Lease lease(*workspace_, stream_);
    if (!input_) input_.reset(max_input_count_);
    set_active_shape(chunk_count, width, stream_);
    const std::size_t input_count =
        static_cast<std::size_t>(chunk_count) * 3U * 48U *
        static_cast<std::size_t>(width);
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(input_.get(), chunk_input,
                                       input_count * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_));
    const bool calculate_probability = score_mode == ScoreMode::kModel;
    launch_compute(input_.get(), stream_, character_policy,
                   calculate_probability, lease);

    const std::size_t rows = static_cast<std::size_t>(active_rows_);
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(host_indices_.get(), indices_.get(),
                                       rows * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_));
    if (calculate_probability) {
      PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(host_prob_.get(), prob_.get(),
                                         rows * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream_));
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (!calculate_probability) {
      std::fill_n(host_prob_.get(), rows, 1.0f);
    }

    append_decoded_predictions(chunk_count, width, return_timesteps, result,
                               std::span<const int>(host_indices_.get(), rows),
                               std::span<const float>(host_prob_.get(), rows),
                               std::span<const float>(chunk_input, input_count),
                               character_policy, score_mode, stream_);
  }

  void run_chunk_device(const float *device_chunk_input, int chunk_count,
                        int width, bool return_timesteps,
                        CharacterPolicy character_policy,
                        ScoreMode score_mode, RecognitionBatchResult &result,
                        cudaStream_t stream) {
    TrtWorkspace::Lease lease(*workspace_, stream);
    set_active_shape(chunk_count, width, stream);
    const bool calculate_probability = score_mode == ScoreMode::kModel;
    launch_compute(device_chunk_input, stream, character_policy,
                   calculate_probability, lease);

    const std::size_t rows = static_cast<std::size_t>(active_rows_);
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(host_indices_.get(), indices_.get(),
                                       rows * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
    if (calculate_probability) {
      PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(host_prob_.get(), prob_.get(),
                                         rows * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream));
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (!calculate_probability) {
      std::fill_n(host_prob_.get(), rows, 1.0f);
    }

    append_decoded_predictions(chunk_count, width, return_timesteps, result,
                               std::span<const int>(host_indices_.get(), rows),
                               std::span<const float>(host_prob_.get(), rows),
                               std::span<const float>{}, character_policy,
                               score_mode, stream);
  }

  void append_decoded_predictions(int chunk_count, int width,
                                  bool return_timesteps,
                                  RecognitionBatchResult &result,
                                  std::span<const int> ids,
                                  std::span<const float> scores,
                                  std::span<const float> host_nchw,
                                  CharacterPolicy character_policy,
                                  ScoreMode score_mode,
                                  cudaStream_t stream) {
    auto decoded = ctc_greedy_decode_batch(
        ids, scores, chunk_count, active_timesteps_, resources_->characters,
        config_.blank_id);

    const bool try_one_stroke_fallback =
        character_policy == CharacterPolicy::kCjkFocusFallback &&
        !host_nchw.empty() &&
        std::any_of(decoded.begin(), decoded.end(),
                    [](const DecodedText &value) {
                      return value.text.empty();
                    });
    if (try_one_stroke_fallback) {
      kernels::cuda_selected_class_vs_max_probability(
          reinterpret_cast<const half *>(hidden_.get()),
          reinterpret_cast<const half *>(resources_->weight.get()),
          reinterpret_cast<const half *>(resources_->bias.get()),
          max_logits_.get(), one_vs_blank_prob_.get(), active_rows_,
          config_.hidden_size, config_.vocab_size, resources_->one_class_id,
          stream);
      const std::size_t rows = static_cast<std::size_t>(active_rows_);
      PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(
          host_one_vs_blank_prob_.get(), one_vs_blank_prob_.get(),
          rows * sizeof(float), cudaMemcpyDeviceToHost, stream));
      PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    result.chunks.push_back(
        RecognitionChunkMeta{chunk_count, width, active_timesteps_});
    result.output_bytes +=
        static_cast<std::size_t>(active_rows_) *
        (sizeof(int) +
         (score_mode == ScoreMode::kModel ? sizeof(float) : 0U));
    for (int row = 0; row < chunk_count; ++row) {
      RecognitionPrediction prediction;
      prediction.decoded = std::move(decoded[static_cast<std::size_t>(row)]);
      if (try_one_stroke_fallback && prediction.decoded.text.empty()) {
        prediction.empty_fallback_attempted = true;
        ++result.empty_fallback_attempted_count;
        const std::size_t start =
            static_cast<std::size_t>(row) *
            static_cast<std::size_t>(active_timesteps_);
        float best_probability = -1.0f;
        for (int timestep = 0; timestep < active_timesteps_; ++timestep) {
          const std::size_t index =
              start + static_cast<std::size_t>(timestep);
          if (ids[index] != config_.blank_id) {
            continue;
          }
          const float probability = host_one_vs_blank_prob_.get()[index];
          if (!std::isfinite(probability) ||
              probability <= best_probability) {
            continue;
          }
          best_probability = probability;
          prediction.empty_fallback_timestep = timestep;
          prediction.empty_fallback_class_id = resources_->one_class_id;
          prediction.empty_fallback_text = kOneToken;
          prediction.empty_fallback_score = probability;
          prediction.empty_fallback_blank_score = 1.0f - probability;
          const float diagnostic_probability =
              std::clamp(probability, 1.0e-12f, 1.0f - 1.0e-7f);
          prediction.empty_fallback_blank_margin =
              std::log((1.0f - diagnostic_probability) /
                       diagnostic_probability);
        }
        const std::size_t image_stride =
            3U * 48U * static_cast<std::size_t>(width);
        const auto features = detect_long_horizontal_stroke(
            host_nchw.subspan(static_cast<std::size_t>(row) * image_stride,
                              image_stride),
            48, width);
        prediction.one_stroke_fallback_evaluated = true;
        prediction.one_stroke_fallback_shape_matched = features.matched;
        prediction.one_stroke_width_ratio = features.width_ratio;
        prediction.one_stroke_aspect_ratio = features.aspect_ratio;
        prediction.one_stroke_height_ratio = features.height_ratio;
        prediction.one_stroke_ink_column_coverage =
            features.ink_column_coverage;
        const bool apply_one_stroke_fallback =
            std::isfinite(prediction.empty_fallback_score) &&
            prediction.empty_fallback_score >=
                kOneStrokeFallbackMinProbability &&
            features.matched;
        if (apply_one_stroke_fallback) {
          prediction.empty_fallback_applied = true;
          prediction.empty_fallback_applied_by = "one_stroke";
          ++result.empty_fallback_applied_count;
          ++result.one_stroke_fallback_applied_count;
          prediction.decoded.text = prediction.empty_fallback_text;
          prediction.decoded.score = prediction.empty_fallback_score;
          prediction.decoded.class_ids = {
              prediction.empty_fallback_class_id};
          prediction.decoded.per_char_scores = {
              prediction.empty_fallback_score};
        }
      }
      if (score_mode == ScoreMode::kAccepted &&
          !prediction.decoded.text.empty()) {
        prediction.decoded.score = 1.0f;
        std::fill(prediction.decoded.per_char_scores.begin(),
                  prediction.decoded.per_char_scores.end(), 1.0f);
      }
      if (return_timesteps) {
        const std::size_t start =
            static_cast<std::size_t>(row) *
            static_cast<std::size_t>(active_timesteps_);
        prediction.timestep_class_ids.assign(
            ids.begin() + static_cast<std::ptrdiff_t>(start),
            ids.begin() + static_cast<std::ptrdiff_t>(start + active_timesteps_));
        if (score_mode == ScoreMode::kAccepted) {
          prediction.timestep_scores.assign(
              static_cast<std::size_t>(active_timesteps_), 1.0f);
        } else {
          prediction.timestep_scores.assign(
              scores.begin() + static_cast<std::ptrdiff_t>(start),
              scores.begin() +
                  static_cast<std::ptrdiff_t>(start + active_timesteps_));
        }
      }
      result.predictions.push_back(std::move(prediction));
    }
  }

  RecognitionWorkerConfig config_;
  std::shared_ptr<SharedResources> resources_;
  std::shared_ptr<TrtWorkspace> workspace_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  CudaPtr<float> input_;
  CudaPtr<std::uint16_t> hidden_;
  CudaPtr<int> indices_;
  CudaPtr<float> prob_;
  CudaPtr<float> max_logits_;
  CudaPtr<float> one_vs_blank_prob_;
  CudaPtr<int> partial_ids_;
  CudaPtr<float> partial_max_;
  CudaPtr<float> partial_sum_;
  CudaHostPtr<int> host_indices_;
  CudaHostPtr<float> host_prob_;
  CudaHostPtr<float> host_one_vs_blank_prob_;
  cudaStream_t stream_ = nullptr;
  nvinfer1::Dims4 input_dims_{};
  nvinfer1::Dims output_dims_{};
  int current_profile_ = -1;
  int active_batch_ = 0;
  int active_width_ = 0;
  int active_timesteps_ = 0;
  int active_rows_ = 0;
  int max_timesteps_ = 0;
  int max_rows_ = 0;
  std::size_t max_input_count_ = 0;
  std::size_t max_hidden_count_ = 0;
  std::size_t context_memory_capacity_bytes_ = 0;
  mutable std::mutex mutex_;
};

RecognitionWorker::RecognitionWorker(RecognitionWorkerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

RecognitionWorker::~RecognitionWorker() = default;

RecognitionBatchResult RecognitionWorker::recognize_f32(
    const float *nchw, int count, int width, int batch_size,
    bool return_timesteps, CharacterPolicy character_policy,
    ScoreMode score_mode) {
  return impl_->recognize_f32(nchw, count, width, batch_size,
                              return_timesteps, character_policy, score_mode);
}

RecognitionBatchResult RecognitionWorker::recognize_device_f32(
    const float *device_nchw, int count, int width, int batch_size,
    bool return_timesteps, CharacterPolicy character_policy,
    ScoreMode score_mode) {
  return impl_->recognize_device_f32(device_nchw, count, width, batch_size,
                                     return_timesteps, character_policy,
                                     score_mode);
}

RecognitionBatchResult RecognitionWorker::recognize_device_f32_on_stream(
    const float *device_nchw, int count, int width, int batch_size,
    bool return_timesteps, cudaStream_t stream,
    CharacterPolicy character_policy, ScoreMode score_mode) {
  return impl_->recognize_device_f32_on_stream(
      device_nchw, count, width, batch_size, return_timesteps, stream,
      character_policy, score_mode);
}

std::string RecognitionWorker::info_json() const { return impl_->info_json(); }

const RecognitionWorkerConfig &RecognitionWorker::config() const noexcept {
  return impl_->config();
}

int RecognitionWorker::classes() const noexcept { return impl_->classes(); }

bool RecognitionWorker::supports_shape(int batch, int width) const {
  return impl_->supports_shape(batch, width);
}

std::string recognition_result_to_json(const RecognitionBatchResult &result,
                                       bool include_timesteps) {
  const double glyphs_per_second =
      result.elapsed_ms > 0.0
          ? static_cast<double>(result.count) / (result.elapsed_ms / 1000.0)
          : 0.0;
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  out << '{';
  out << "\"count\":" << result.count;
  out << ",\"width\":" << result.width;
  out << ",\"batch_size\":" << result.batch_size;
  out << ",\"elapsed_ms\":" << result.elapsed_ms;
  out << ",\"glyphs_per_second\":" << glyphs_per_second;
  out << ",\"input_shapes\":[";
  for (std::size_t i = 0; i < result.chunks.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto &chunk = result.chunks[i];
    out << '[' << chunk.batch << ",3,48," << chunk.width << ']';
  }
  out << ']';
  out << ",\"output_shapes\":[";
  for (std::size_t i = 0; i < result.chunks.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto &chunk = result.chunks[i];
    out << "[[" << chunk.batch << ',' << chunk.timesteps << "],["
        << chunk.batch << ',' << chunk.timesteps << "]]";
  }
  out << ']';
  out << ",\"output_bytes\":" << result.output_bytes;
  const char *model_score_type =
      result.character_policy == CharacterPolicy::kAll
          ? "probability"
          : "conditional_probability";
  const char *score_type = result.score_mode == ScoreMode::kAccepted
                               ? "binary_acceptance"
                               : model_score_type;
  out << ",\"character_policy\":\""
      << character_policy_name(result.character_policy) << '"';
  out << ",\"score_mode\":\"" << score_mode_name(result.score_mode) << '"';
  out << ",\"score_type\":\"" << score_type << '"';
  out << ",\"model_score_type\":\"" << model_score_type << '"';
  out << ",\"empty_fallback_attempted_count\":"
      << result.empty_fallback_attempted_count;
  out << ",\"empty_fallback_applied_count\":"
      << result.empty_fallback_applied_count;
  out << ",\"one_stroke_fallback_applied_count\":"
      << result.one_stroke_fallback_applied_count;
  if (result.character_policy == CharacterPolicy::kCjkFocusFallback) {
    out << ",\"one_stroke_fallback_min_probability\":"
        << kOneStrokeFallbackMinProbability;
  }

  out << ",\"meta\":{";
  out << "\"predict_elapsed_ms\":" << result.elapsed_ms;
  out << ",\"glyphs_per_second\":" << glyphs_per_second;
  out << ",\"output_bytes\":" << result.output_bytes;
  out << ",\"count\":" << result.count;
  out << ",\"width\":" << result.width;
  out << ",\"batch_size\":" << result.batch_size;
  out << ",\"character_policy\":\""
      << character_policy_name(result.character_policy) << '"';
  out << ",\"score_mode\":\"" << score_mode_name(result.score_mode) << '"';
  out << ",\"score_type\":\"" << score_type << '"';
  out << ",\"model_score_type\":\"" << model_score_type << '"';
  out << ",\"empty_fallback_attempted_count\":"
      << result.empty_fallback_attempted_count;
  out << ",\"empty_fallback_applied_count\":"
      << result.empty_fallback_applied_count;
  out << ",\"one_stroke_fallback_applied_count\":"
      << result.one_stroke_fallback_applied_count;
  if (result.character_policy == CharacterPolicy::kCjkFocusFallback) {
    out << ",\"one_stroke_fallback_min_probability\":"
        << kOneStrokeFallbackMinProbability;
  }
  out << '}';

  out << ",\"predictions\":[";
  for (std::size_t i = 0; i < result.predictions.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto &prediction = result.predictions[i];
    out << '{';
    out << "\"text\":";
    append_json_escaped(out, prediction.decoded.text);
    out << ",\"score\":";
    append_float(out, prediction.decoded.score);
    out << ",\"class_ids\":";
    append_int_array(out, prediction.decoded.class_ids);
    out << ",\"per_char_scores\":";
    append_float_array(out, prediction.decoded.per_char_scores);
    if (prediction.empty_fallback_attempted) {
      out << ",\"empty_fallback\":{";
      out << "\"applied\":"
          << (prediction.empty_fallback_applied ? "true" : "false");
      out << ",\"applied_by\":";
      if (prediction.empty_fallback_applied_by.empty()) {
        out << "null";
      } else {
        append_json_escaped(out, prediction.empty_fallback_applied_by);
      }
      out << ",\"timestep\":" << prediction.empty_fallback_timestep;
      out << ",\"class_id\":" << prediction.empty_fallback_class_id;
      out << ",\"text\":";
      append_json_escaped(out, prediction.empty_fallback_text);
      out << ",\"score\":";
      append_float(out, prediction.empty_fallback_score);
      out << ",\"score_type\":\"one_vs_blank_probability\"";
      out << ",\"blank_score\":";
      append_float(out, prediction.empty_fallback_blank_score);
      out << ",\"blank_margin\":";
      append_float(out, prediction.empty_fallback_blank_margin);
      if (prediction.one_stroke_fallback_evaluated) {
        out << ",\"one_stroke\":{";
        out << "\"shape_matched\":"
            << (prediction.one_stroke_fallback_shape_matched ? "true"
                                                            : "false");
        out << ",\"width_ratio\":";
        append_float(out, prediction.one_stroke_width_ratio);
        out << ",\"aspect_ratio\":";
        append_float(out, prediction.one_stroke_aspect_ratio);
        out << ",\"height_ratio\":";
        append_float(out, prediction.one_stroke_height_ratio);
        out << ",\"ink_column_coverage\":";
        append_float(out, prediction.one_stroke_ink_column_coverage);
        out << '}';
      }
      out << '}';
    }
    if (include_timesteps) {
      out << ",\"timestep_class_ids\":";
      append_int_array(out, prediction.timestep_class_ids);
      out << ",\"timestep_scores\":";
      append_float_array(out, prediction.timestep_scores);
    }
    out << '}';
  }
  out << ']';
  out << '}';
  return out.str();
}

} // namespace ppocrv6_native::recognition
