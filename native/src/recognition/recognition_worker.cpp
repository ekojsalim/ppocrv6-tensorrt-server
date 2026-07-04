#include "ppocrv6_native/recognition/recognition_worker.h"

#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/engine/trt_engine.h"
#include "ppocrv6_native/kernels/classifier.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppocrv6_native::recognition {
namespace {

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

int resolve_profile(nvinfer1::ICudaEngine &engine,
                    const std::string &input_name, int batch, int width) {
  const nvinfer1::Dims4 dims{batch, 3, 48, width};
  const int profiles = engine.getNbOptimizationProfiles();
  for (int profile = 0; profile < profiles; ++profile) {
    const auto min_dims = engine.getProfileShape(
        input_name.c_str(), profile, nvinfer1::OptProfileSelector::kMIN);
    const auto max_dims = engine.getProfileShape(
        input_name.c_str(), profile, nvinfer1::OptProfileSelector::kMAX);
    if (dims_contains(min_dims, max_dims, dims)) {
      return profile;
    }
  }
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

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("failed to create TensorRT runtime");
    }
    const auto engine_bytes = read_file(config_.engine_path);
    engine_.reset(
        runtime_->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!engine_) {
      throw std::runtime_error("failed to deserialize engine " +
                               config_.engine_path);
    }
    input_name_ = discover_input_name(*engine_);
    output_name_ = discover_output_name(*engine_);

    const std::size_t weight_count =
        static_cast<std::size_t>(config_.hidden_size) *
        static_cast<std::size_t>(config_.vocab_size);
    const std::size_t bias_count = static_cast<std::size_t>(config_.vocab_size);
    const auto host_weight =
        read_binary<std::uint16_t>(config_.weight_path, weight_count);
    const auto host_bias =
        read_binary<std::uint16_t>(config_.bias_path, bias_count);
    characters_ = read_characters(config_.characters_path);

    weight_.reset(weight_count);
    bias_.reset(bias_count);
    PPOCRV6_CUDA_CHECK(cudaMemcpy(weight_.get(), host_weight.data(),
                                  host_weight.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice));
    PPOCRV6_CUDA_CHECK(cudaMemcpy(bias_.get(), host_bias.data(),
                                  host_bias.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice));

    context_.reset(engine_->createExecutionContext(
        nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!context_) {
      throw std::runtime_error("failed to create TensorRT execution context");
    }
    context_memory_bytes_ =
        static_cast<std::size_t>(std::max<int64_t>(0, engine_->getDeviceMemorySizeV2()));
    context_memory_.reset(context_memory_bytes_);
    if (context_memory_bytes_ > 0) {
      context_->setDeviceMemoryV2(
          context_memory_.get(), static_cast<int64_t>(context_memory_bytes_));
    }

    PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream_));
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
                                       bool return_timesteps) {
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
    result.predictions.reserve(static_cast<std::size_t>(count));
    const auto started = std::chrono::steady_clock::now();

    const std::size_t image_stride =
        3U * 48U * static_cast<std::size_t>(width);
    for (int start = 0; start < count; start += batch_size) {
      const int chunk_count = std::min(batch_size, count - start);
      run_chunk(nchw + static_cast<std::size_t>(start) * image_stride,
                chunk_count, width, return_timesteps, result);
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));

    result.elapsed_ms =
        elapsed_ms(started, std::chrono::steady_clock::now());
    return result;
  }

  RecognitionBatchResult recognize_device_f32(const float *device_nchw,
                                              int count, int width,
                                              int batch_size,
                                              bool return_timesteps) {
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
    result.predictions.reserve(static_cast<std::size_t>(count));
    const auto started = std::chrono::steady_clock::now();

    const std::size_t image_stride =
        3U * 48U * static_cast<std::size_t>(width);
    for (int start = 0; start < count; start += batch_size) {
      const int chunk_count = std::min(batch_size, count - start);
      run_chunk_device(device_nchw +
                           static_cast<std::size_t>(start) * image_stride,
                       chunk_count, width, return_timesteps, result);
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));

    result.elapsed_ms =
        elapsed_ms(started, std::chrono::steady_clock::now());
    return result;
  }

  [[nodiscard]] std::string info_json() const {
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
    out << ",\"classes\":" << characters_.size();
    out << ",\"default_width\":" << config_.default_width;
    out << ",\"default_batch_size\":" << config_.default_batch_size;
    out << ",\"max_width\":" << config_.max_width;
    out << ",\"max_batch_size\":" << config_.max_batch_size;
    out << ",\"vocab_size\":" << config_.vocab_size;
    out << ",\"hidden_size\":" << config_.hidden_size;
    out << ",\"vocab_tile_size\":" << config_.vocab_tile_size;
    out << ",\"score_type\":\"probability\"";
    out << ",\"context_mode\":\"switch\"";
    out << ",\"profiles\":" << (engine_ ? engine_->getNbOptimizationProfiles() : 0);
    out << '}';
    return out.str();
  }

  [[nodiscard]] const RecognitionWorkerConfig &config() const noexcept {
    return config_;
  }

  [[nodiscard]] int classes() const noexcept {
    return static_cast<int>(characters_.size());
  }

private:
  void allocate_for_max_shape() {
    set_active_shape(config_.max_batch_size, config_.max_width);
    max_timesteps_ = output_dims_.d[1];
    max_rows_ = config_.max_batch_size * max_timesteps_;
    max_input_count_ =
        static_cast<std::size_t>(config_.max_batch_size) * 3U * 48U *
        static_cast<std::size_t>(config_.max_width);
    max_hidden_count_ = volume(output_dims_);
    const auto workspace = kernels::tiled_classifier_workspace_shape(
        max_rows_, config_.vocab_size, 16, config_.vocab_tile_size);

    input_.reset(max_input_count_);
    hidden_.reset(max_hidden_count_);
    indices_.reset(static_cast<std::size_t>(max_rows_));
    prob_.reset(static_cast<std::size_t>(max_rows_));
    max_logits_.reset(static_cast<std::size_t>(max_rows_));
    partial_ids_.reset(workspace.partial_count);
    partial_max_.reset(workspace.partial_count);
    partial_sum_.reset(workspace.partial_count);
  }

  void warmup() {
    if (config_.warmup_runs <= 0) {
      return;
    }
    const int warmup_batch =
        std::min(config_.default_batch_size, config_.max_batch_size);
    set_active_shape(warmup_batch, config_.default_width);
    const std::size_t input_count =
        static_cast<std::size_t>(warmup_batch) * 3U * 48U *
        static_cast<std::size_t>(config_.default_width);
    PPOCRV6_CUDA_CHECK(
        cudaMemsetAsync(input_.get(), 0, input_count * sizeof(float), stream_));
    for (int i = 0; i < config_.warmup_runs; ++i) {
      launch_compute();
    }
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  void set_active_shape(int batch, int width) {
    const int profile = resolve_profile(*engine_, input_name_, batch, width);
    if (profile != current_profile_) {
      if (!context_->setOptimizationProfileAsync(profile, stream_)) {
        throw std::runtime_error("failed to set TensorRT optimization profile " +
                                 std::to_string(profile));
      }
      current_profile_ = profile;
    }
    input_dims_ = nvinfer1::Dims4{batch, 3, 48, width};
    if (!context_->setInputShape(input_name_.c_str(), input_dims_)) {
      throw std::runtime_error("failed to set TensorRT input shape " +
                               ppocrv6_native::engine::dims_to_string(
                                   input_dims_));
    }
    output_dims_ = context_->getTensorShape(output_name_.c_str());
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
    if (!context_->setTensorAddress(input_name_.c_str(), input_.get()) ||
        !context_->setTensorAddress(output_name_.c_str(), hidden_.get())) {
      throw std::runtime_error("failed to set TensorRT tensor addresses");
    }
  }

  void launch_compute() {
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT hidden enqueue failed");
    }
    kernels::cuda_linear_argmax_prob_wmma(
        reinterpret_cast<const half *>(hidden_.get()),
        reinterpret_cast<const half *>(weight_.get()),
        reinterpret_cast<const half *>(bias_.get()), indices_.get(),
        prob_.get(), max_logits_.get(), active_rows_, config_.hidden_size,
        config_.vocab_size, config_.vocab_tile_size, partial_ids_.get(),
        partial_max_.get(), partial_sum_.get(), stream_);
  }

  void run_chunk(const float *chunk_input, int chunk_count, int width,
                 bool return_timesteps, RecognitionBatchResult &result) {
    set_active_shape(chunk_count, width);
    const std::size_t input_count =
        static_cast<std::size_t>(chunk_count) * 3U * 48U *
        static_cast<std::size_t>(width);
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(input_.get(), chunk_input,
                                       input_count * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_));
    launch_compute();

    const std::size_t rows = static_cast<std::size_t>(active_rows_);
    std::vector<int> ids(rows);
    std::vector<float> scores(rows);
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(ids.data(), indices_.get(),
                                       ids.size() * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_));
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(scores.data(), prob_.get(),
                                       scores.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_));
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));

    append_decoded_predictions(chunk_count, width, return_timesteps, result,
                               ids, scores);
  }

  void run_chunk_device(const float *device_chunk_input, int chunk_count,
                        int width, bool return_timesteps,
                        RecognitionBatchResult &result) {
    set_active_shape(chunk_count, width);
    const std::size_t input_count =
        static_cast<std::size_t>(chunk_count) * 3U * 48U *
        static_cast<std::size_t>(width);
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(input_.get(), device_chunk_input,
                                       input_count * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream_));
    launch_compute();

    const std::size_t rows = static_cast<std::size_t>(active_rows_);
    std::vector<int> ids(rows);
    std::vector<float> scores(rows);
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(ids.data(), indices_.get(),
                                       ids.size() * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream_));
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(scores.data(), prob_.get(),
                                       scores.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream_));
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream_));

    append_decoded_predictions(chunk_count, width, return_timesteps, result,
                               ids, scores);
  }

  void append_decoded_predictions(int chunk_count, int width,
                                  bool return_timesteps,
                                  RecognitionBatchResult &result,
                                  const std::vector<int> &ids,
                                  const std::vector<float> &scores) {
    auto decoded = ctc_greedy_decode_batch(
        std::span<const int>(ids.data(), ids.size()),
        std::span<const float>(scores.data(), scores.size()), chunk_count,
        active_timesteps_, characters_, config_.blank_id);

    result.chunks.push_back(
        RecognitionChunkMeta{chunk_count, width, active_timesteps_});
    result.output_bytes +=
        static_cast<std::size_t>(active_rows_) * (sizeof(int) + sizeof(float));
    for (int row = 0; row < chunk_count; ++row) {
      RecognitionPrediction prediction;
      prediction.decoded = std::move(decoded[static_cast<std::size_t>(row)]);
      if (return_timesteps) {
        const std::size_t start =
            static_cast<std::size_t>(row) *
            static_cast<std::size_t>(active_timesteps_);
        prediction.timestep_class_ids.assign(
            ids.begin() + static_cast<std::ptrdiff_t>(start),
            ids.begin() + static_cast<std::ptrdiff_t>(start + active_timesteps_));
        prediction.timestep_scores.assign(
            scores.begin() + static_cast<std::ptrdiff_t>(start),
            scores.begin() +
                static_cast<std::ptrdiff_t>(start + active_timesteps_));
      }
      result.predictions.push_back(std::move(prediction));
    }
  }

  RecognitionWorkerConfig config_;
  std::vector<std::string> characters_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  CudaPtr<std::uint8_t> context_memory_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  CudaPtr<float> input_;
  CudaPtr<std::uint16_t> hidden_;
  CudaPtr<std::uint16_t> weight_;
  CudaPtr<std::uint16_t> bias_;
  CudaPtr<int> indices_;
  CudaPtr<float> prob_;
  CudaPtr<float> max_logits_;
  CudaPtr<int> partial_ids_;
  CudaPtr<float> partial_max_;
  CudaPtr<float> partial_sum_;
  cudaStream_t stream_ = nullptr;
  std::string input_name_;
  std::string output_name_;
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
  std::size_t context_memory_bytes_ = 0;
  std::mutex mutex_;
};

RecognitionWorker::RecognitionWorker(RecognitionWorkerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

RecognitionWorker::~RecognitionWorker() = default;

RecognitionBatchResult RecognitionWorker::recognize_f32(
    const float *nchw, int count, int width, int batch_size,
    bool return_timesteps) {
  return impl_->recognize_f32(nchw, count, width, batch_size,
                              return_timesteps);
}

RecognitionBatchResult RecognitionWorker::recognize_device_f32(
    const float *device_nchw, int count, int width, int batch_size,
    bool return_timesteps) {
  return impl_->recognize_device_f32(device_nchw, count, width, batch_size,
                                     return_timesteps);
}

std::string RecognitionWorker::info_json() const { return impl_->info_json(); }

const RecognitionWorkerConfig &RecognitionWorker::config() const noexcept {
  return impl_->config();
}

int RecognitionWorker::classes() const noexcept { return impl_->classes(); }

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
  out << ",\"score_type\":\"probability\"";

  out << ",\"meta\":{";
  out << "\"predict_elapsed_ms\":" << result.elapsed_ms;
  out << ",\"glyphs_per_second\":" << glyphs_per_second;
  out << ",\"output_bytes\":" << result.output_bytes;
  out << ",\"count\":" << result.count;
  out << ",\"width\":" << result.width;
  out << ",\"batch_size\":" << result.batch_size;
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
