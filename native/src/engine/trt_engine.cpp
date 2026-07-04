#include "ppocrv6_native/engine/trt_engine.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

namespace ppocrv6_native::engine {

void Logger::log(Severity severity, const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
    std::cerr << "[TRT] " << msg << '\n';
  }
}

TrtEngine::TrtEngine(std::string engine_path)
    : engine_path_(std::move(engine_path)) {}

bool TrtEngine::load() {
  std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "[TRT] failed to open engine: " << engine_path_ << '\n';
    return false;
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    std::cerr << "[TRT] invalid engine size: " << engine_path_ << '\n';
    return false;
  }
  std::vector<char> bytes(static_cast<std::size_t>(size));
  file.seekg(0, std::ios::beg);
  file.read(bytes.data(), size);
  if (!file) {
    std::cerr << "[TRT] failed to read engine: " << engine_path_ << '\n';
    return false;
  }

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    std::cerr << "[TRT] failed to create runtime\n";
    return false;
  }
  engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!engine_) {
    std::cerr << "[TRT] failed to deserialize engine\n";
    return false;
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    std::cerr << "[TRT] failed to create execution context\n";
    return false;
  }

  input_names_.clear();
  output_names_.clear();
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char *name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_names_.emplace_back(name);
    } else {
      output_names_.emplace_back(name);
    }
  }
  return true;
}

bool TrtEngine::set_input_shape(const std::string &name,
                                const nvinfer1::Dims &dims) {
  if (!context_) {
    return false;
  }
  if (!context_->setInputShape(name.c_str(), dims)) {
    std::cerr << "[TRT] setInputShape failed for " << name << " dims="
              << dims_to_string(dims) << '\n';
    return false;
  }
  return true;
}

void TrtEngine::set_tensor_address(const std::string &name, void *ptr) {
  if (context_) {
    context_->setTensorAddress(name.c_str(), ptr);
  }
}

bool TrtEngine::execute(cudaStream_t stream) {
  if (!context_) {
    return false;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr << "[TRT] not all input dimensions are specified\n";
    return false;
  }
  return context_->enqueueV3(stream);
}

nvinfer1::Dims TrtEngine::tensor_shape(const std::string &name) const {
  if (!context_) {
    return {};
  }
  return context_->getTensorShape(name.c_str());
}

const std::vector<std::string> &TrtEngine::input_names() const noexcept {
  return input_names_;
}

const std::vector<std::string> &TrtEngine::output_names() const noexcept {
  return output_names_;
}

int TrtEngine::num_profiles() const noexcept {
  return engine_ ? engine_->getNbOptimizationProfiles() : 0;
}

std::string dims_to_string(const nvinfer1::Dims &dims) {
  std::ostringstream out;
  out << '[';
  for (int i = 0; i < dims.nbDims; ++i) {
    if (i != 0) {
      out << ',';
    }
    out << dims.d[i];
  }
  out << ']';
  return out.str();
}

} // namespace ppocrv6_native::engine
