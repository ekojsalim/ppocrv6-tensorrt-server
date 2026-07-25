#include "ppocrv6_native/detection/detection_worker.h"

#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/engine/trt_engine.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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

namespace ppocrv6_native::detection {
namespace {

class Logger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[ppocrv6-detector] " << msg << '\n';
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

std::size_t dtype_size(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
  case nvinfer1::DataType::kBF16:
    return 2;
  case nvinfer1::DataType::kINT64:
    return 8;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kINT8:
  case nvinfer1::DataType::kBOOL:
  case nvinfer1::DataType::kUINT8:
  case nvinfer1::DataType::kFP8:
  case nvinfer1::DataType::kE8M0:
  case nvinfer1::DataType::kINT4:
  case nvinfer1::DataType::kFP4:
    return 1;
  }
  throw std::runtime_error("unsupported TensorRT data type");
}

std::string dtype_name(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return "float32";
  case nvinfer1::DataType::kHALF:
    return "float16";
  case nvinfer1::DataType::kBF16:
    return "bf16";
  case nvinfer1::DataType::kINT64:
    return "int64";
  case nvinfer1::DataType::kINT32:
    return "int32";
  case nvinfer1::DataType::kINT8:
    return "int8";
  case nvinfer1::DataType::kBOOL:
    return "bool";
  case nvinfer1::DataType::kUINT8:
    return "uint8";
  case nvinfer1::DataType::kFP8:
    return "fp8";
  case nvinfer1::DataType::kE8M0:
    return "e8m0";
  case nvinfer1::DataType::kINT4:
    return "int4";
  case nvinfer1::DataType::kFP4:
    return "fp4";
  }
  return "unknown";
}

std::string discover_tensor_name(nvinfer1::ICudaEngine &engine,
                                 nvinfer1::TensorIOMode mode,
                                 const char *description) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char *name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) == mode) {
      return name;
    }
  }
  throw std::runtime_error(std::string("failed to discover TensorRT ") +
                           description + " name");
}

int resolve_profile(nvinfer1::ICudaEngine &engine,
                    const std::string &input_name, int batch, int height,
                    int width) {
  const nvinfer1::Dims4 dims{batch, 3, height, width};
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
  throw std::runtime_error("no TensorRT detector profile accepts shape " +
                           ppocrv6_native::engine::dims_to_string(dims));
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t stop) {
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

} // namespace

class DetectionWorker::Impl {
public:
  explicit Impl(DetectionWorkerConfig config) : config_(std::move(config)) {
    if (config_.default_batch <= 0 || config_.default_height <= 0 ||
        config_.default_width <= 0 || config_.max_batch <= 0 ||
        config_.max_height <= 0 || config_.max_width <= 0) {
      throw std::runtime_error("invalid detection worker config");
    }
    if (config_.default_batch > config_.max_batch ||
        config_.default_height > config_.max_height ||
        config_.default_width > config_.max_width) {
      throw std::runtime_error("default detection shape exceeds max shape");
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

    input_name_ = discover_tensor_name(*engine_, nvinfer1::TensorIOMode::kINPUT,
                                       "input");
    output_name_ = discover_tensor_name(
        *engine_, nvinfer1::TensorIOMode::kOUTPUT, "output");
    input_dtype_ = engine_->getTensorDataType(input_name_.c_str());
    output_dtype_ = engine_->getTensorDataType(output_name_.c_str());
    if (input_dtype_ != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("detector input dtype is not float32");
    }
    if (output_dtype_ != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("detector output dtype is not float32");
    }

    context_.reset(engine_->createExecutionContext(
        nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!context_) {
      throw std::runtime_error("failed to create detector execution context");
    }
    context_memory_bytes_ =
        static_cast<std::size_t>(std::max<int64_t>(0, engine_->getDeviceMemorySizeV2()));
    context_memory_.reset(context_memory_bytes_);
    if (context_memory_bytes_ > 0) {
      context_->setDeviceMemoryV2(
          context_memory_.get(), static_cast<int64_t>(context_memory_bytes_));
    }

    PPOCRV6_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&event_start_));
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&event_after_input_));
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&event_after_execute_));
    PPOCRV6_CUDA_CHECK(cudaEventCreate(&event_after_output_));
    allocate_for_max_shape();
    warmup();
  }

  ~Impl() {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (event_start_ != nullptr) {
      cudaEventDestroy(event_start_);
    }
    if (event_after_input_ != nullptr) {
      cudaEventDestroy(event_after_input_);
    }
    if (event_after_execute_ != nullptr) {
      cudaEventDestroy(event_after_execute_);
    }
    if (event_after_output_ != nullptr) {
      cudaEventDestroy(event_after_output_);
    }
  }

  DetectionTensorResult detect_f32(const float *nchw, int batch, int height,
                                   int width, bool copy_output) {
    if (nchw == nullptr) {
      throw std::runtime_error("input tensor pointer is null");
    }
    if (batch <= 0 || batch > config_.max_batch || height <= 0 ||
        height > config_.max_height || width <= 0 || width > config_.max_width) {
      throw std::runtime_error("detector input shape is outside worker limits");
    }

    std::lock_guard<std::mutex> guard(mutex_);
    const auto started = std::chrono::steady_clock::now();
    DetectionTensorResult result;
    result.batch = batch;
    result.height = height;
    result.width = width;

    run_once(nchw, false, batch, height, width, copy_output, nullptr, result);
    result.elapsed_ms = elapsed_ms(started, std::chrono::steady_clock::now());
    return result;
  }

  DetectionTensorResult detect_device_f32(const float *device_nchw, int batch,
                                          int height, int width,
                                          bool copy_output,
                                          cudaEvent_t input_ready) {
    if (device_nchw == nullptr) {
      throw std::runtime_error("device input tensor pointer is null");
    }
    if (batch <= 0 || batch > config_.max_batch || height <= 0 ||
        height > config_.max_height || width <= 0 || width > config_.max_width) {
      throw std::runtime_error("detector input shape is outside worker limits");
    }

    std::lock_guard<std::mutex> guard(mutex_);
    const auto started = std::chrono::steady_clock::now();
    DetectionTensorResult result;
    result.batch = batch;
    result.height = height;
    result.width = width;
    run_once(device_nchw, true, batch, height, width, copy_output, input_ready,
             result);
    result.elapsed_ms = elapsed_ms(started, std::chrono::steady_clock::now());
    return result;
  }

  [[nodiscard]] std::string info_json() const {
    std::ostringstream out;
    out << '{';
    out << "\"engine\":";
    append_json_escaped(out, config_.engine_path);
    out << ",\"input_name\":";
    append_json_escaped(out, input_name_);
    out << ",\"output_name\":";
    append_json_escaped(out, output_name_);
    out << ",\"input_dtype\":";
    append_json_escaped(out, dtype_name(input_dtype_));
    out << ",\"output_dtype\":";
    append_json_escaped(out, dtype_name(output_dtype_));
    out << ",\"providers\":[\"TensorRT\",\"CUDA\"]";
    out << ",\"default_shape\":[" << config_.default_batch << ",3,"
        << config_.default_height << ',' << config_.default_width << ']';
    out << ",\"max_shape\":[" << config_.max_batch << ",3,"
        << config_.max_height << ',' << config_.max_width << ']';
    out << ",\"profiles\":" << engine_->getNbOptimizationProfiles();
    out << ",\"context_memory_bytes\":" << context_memory_bytes_;
    out << ",\"context_memory_mib\":";
    append_float(out, static_cast<double>(context_memory_bytes_) /
                          (1024.0 * 1024.0));
    out << '}';
    return out.str();
  }

  [[nodiscard]] const DetectionWorkerConfig &config() const noexcept {
    return config_;
  }

private:
  void allocate_for_max_shape() {
    max_input_count_ = static_cast<std::size_t>(config_.max_batch) * 3U *
                       static_cast<std::size_t>(config_.max_height) *
                       static_cast<std::size_t>(config_.max_width);
    max_output_count_ = static_cast<std::size_t>(config_.max_batch) *
                        static_cast<std::size_t>(config_.max_height) *
                        static_cast<std::size_t>(config_.max_width);
    device_input_.reset(max_input_count_);
    device_output_.reset(max_output_count_);
    host_output_.reset(max_output_count_);
  }

  void warmup() {
    if (config_.warmup_runs <= 0) {
      return;
    }
    const std::size_t input_count =
        static_cast<std::size_t>(config_.default_batch) * 3U *
        static_cast<std::size_t>(config_.default_height) *
        static_cast<std::size_t>(config_.default_width);
    std::vector<float> zeros(input_count, 0.0f);
    for (int i = 0; i < config_.warmup_runs; ++i) {
      (void)detect_f32(zeros.data(), config_.default_batch,
                       config_.default_height, config_.default_width, false);
    }
  }

  void run_once(const float *nchw, bool input_is_device, int batch, int height,
                int width, bool copy_output, cudaEvent_t input_ready,
                DetectionTensorResult &result) {
    const std::size_t input_count = static_cast<std::size_t>(batch) * 3U *
                                    static_cast<std::size_t>(height) *
                                    static_cast<std::size_t>(width);
    if (input_count > max_input_count_) {
      throw std::runtime_error("detector input exceeds allocated buffer");
    }

    const int profile =
        resolve_profile(*engine_, input_name_, batch, height, width);
    if (active_profile_ != profile) {
      if (!context_->setOptimizationProfileAsync(profile, stream_)) {
        throw std::runtime_error("failed to set detector optimization profile");
      }
      active_profile_ = profile;
    }

    const nvinfer1::Dims4 input_dims{batch, 3, height, width};
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
      throw std::runtime_error("failed to set detector input shape " +
                               ppocrv6_native::engine::dims_to_string(input_dims));
    }
    const auto output_dims = context_->getTensorShape(output_name_.c_str());
    const std::size_t output_count = volume(output_dims);
    if (output_count == 0 || output_count > max_output_count_) {
      throw std::runtime_error("invalid detector output shape " +
                               ppocrv6_native::engine::dims_to_string(output_dims));
    }

    result.output_n = output_dims.nbDims > 0 ? output_dims.d[0] : 0;
    result.output_c = output_dims.nbDims > 1 ? output_dims.d[1] : 0;
    result.output_h = output_dims.nbDims > 2 ? output_dims.d[2] : 0;
    result.output_w = output_dims.nbDims > 3 ? output_dims.d[3] : 0;
    result.output_bytes = output_count * dtype_size(output_dtype_);

    if (input_ready != nullptr) {
      PPOCRV6_CUDA_CHECK(cudaStreamWaitEvent(stream_, input_ready, 0));
    }
    PPOCRV6_CUDA_CHECK(cudaEventRecord(event_start_, stream_));
    const float *device_input = nchw;
    if (!input_is_device) {
      PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(
          device_input_.get(), nchw, input_count * sizeof(float),
          cudaMemcpyHostToDevice, stream_));
      device_input = device_input_.get();
    }
    PPOCRV6_CUDA_CHECK(cudaEventRecord(event_after_input_, stream_));

    if (!context_->setTensorAddress(input_name_.c_str(),
                                    const_cast<float *>(device_input)) ||
        !context_->setTensorAddress(output_name_.c_str(), device_output_.get())) {
      throw std::runtime_error("failed to set detector tensor addresses");
    }

    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT detector enqueue failed");
    }
    PPOCRV6_CUDA_CHECK(cudaEventRecord(event_after_execute_, stream_));

    if (copy_output) {
      PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(host_output_.get(),
                                         device_output_.get(),
                                         output_count * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream_));
    }
    PPOCRV6_CUDA_CHECK(cudaEventRecord(event_after_output_, stream_));
    PPOCRV6_CUDA_CHECK(cudaEventSynchronize(event_after_output_));
    result.h2d_ms =
        input_is_device
            ? 0.0f
            : elapsed_event_ms(event_start_, event_after_input_);
    result.execute_ms =
        elapsed_event_ms(event_after_input_, event_after_execute_);
    result.d2h_ms =
        copy_output
            ? elapsed_event_ms(event_after_execute_, event_after_output_)
            : 0.0f;
    if (copy_output) {
      result.output.resize(output_count);
      std::memcpy(result.output.data(), host_output_.get(),
                  output_count * sizeof(float));
    }
  }

  DetectionWorkerConfig config_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::string input_name_;
  std::string output_name_;
  nvinfer1::DataType input_dtype_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_dtype_ = nvinfer1::DataType::kFLOAT;
  std::size_t context_memory_bytes_ = 0;
  CudaPtr<unsigned char> context_memory_;
  CudaPtr<float> device_input_;
  CudaPtr<float> device_output_;
  CudaHostPtr<float> host_output_;
  std::size_t max_input_count_ = 0;
  std::size_t max_output_count_ = 0;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_start_ = nullptr;
  cudaEvent_t event_after_input_ = nullptr;
  cudaEvent_t event_after_execute_ = nullptr;
  cudaEvent_t event_after_output_ = nullptr;
  int active_profile_ = -1;
  mutable std::mutex mutex_;
};

DetectionWorker::DetectionWorker(DetectionWorkerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

DetectionWorker::~DetectionWorker() = default;

DetectionTensorResult DetectionWorker::detect_f32(const float *nchw, int batch,
                                                  int height, int width,
                                                  bool copy_output) {
  return impl_->detect_f32(nchw, batch, height, width, copy_output);
}

DetectionTensorResult DetectionWorker::detect_device_f32(
    const float *device_nchw, int batch, int height, int width,
    bool copy_output, cudaEvent_t input_ready) {
  return impl_->detect_device_f32(device_nchw, batch, height, width, copy_output,
                                  input_ready);
}

std::string DetectionWorker::info_json() const { return impl_->info_json(); }

const DetectionWorkerConfig &DetectionWorker::config() const noexcept {
  return impl_->config();
}

std::string detection_result_to_json(const DetectionTensorResult &result) {
  std::ostringstream out;
  out << '{';
  out << "\"input_shape\":[" << result.batch << ",3," << result.height << ','
      << result.width << ']';
  out << ",\"output_shape\":[" << result.output_n << ',' << result.output_c
      << ',' << result.output_h << ',' << result.output_w << ']';
  out << ",\"elapsed_ms\":";
  append_float(out, result.elapsed_ms);
  out << ",\"h2d_ms\":";
  append_float(out, result.h2d_ms);
  out << ",\"execute_ms\":";
  append_float(out, result.execute_ms);
  out << ",\"d2h_ms\":";
  append_float(out, result.d2h_ms);
  out << ",\"output_bytes\":" << result.output_bytes;
  out << '}';
  return out.str();
}

} // namespace ppocrv6_native::detection
