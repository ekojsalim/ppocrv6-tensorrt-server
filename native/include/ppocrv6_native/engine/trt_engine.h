#pragma once

#include <NvInfer.h>

#include <memory>
#include <string>
#include <vector>

namespace ppocrv6_native::engine {

class Logger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override;
};

class TrtEngine {
public:
  explicit TrtEngine(std::string engine_path);

  bool load();
  bool set_input_shape(const std::string &name, const nvinfer1::Dims &dims);
  void set_tensor_address(const std::string &name, void *ptr);
  bool execute(cudaStream_t stream = nullptr);

  [[nodiscard]] nvinfer1::Dims tensor_shape(const std::string &name) const;
  [[nodiscard]] const std::vector<std::string> &input_names() const noexcept;
  [[nodiscard]] const std::vector<std::string> &output_names() const noexcept;
  [[nodiscard]] int num_profiles() const noexcept;

private:
  std::string engine_path_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

std::string dims_to_string(const nvinfer1::Dims &dims);

} // namespace ppocrv6_native::engine
