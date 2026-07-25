#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/engine/onnx_to_trt.h"
#include "ppocrv6_native/engine/trt_engine.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Logger final : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT multi-profile] " << msg << '\n';
    }
  }
};

struct RunResult {
  int profile = 0;
  int batch = 0;
  int width = 0;
  nvinfer1::Dims output_dims{};
  float enqueue_ms = 0.0f;
  std::size_t input_bytes = 0;
  std::size_t output_bytes = 0;
};

std::map<std::string, std::string> parse_args(int argc, char **argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) {
      throw std::runtime_error("unexpected positional argument: " + key);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + key);
    }
    args[key] = argv[++i];
  }
  return args;
}

std::string arg_or(const std::map<std::string, std::string> &args,
                   const std::string &key, const std::string &fallback) {
  const auto it = args.find(key);
  return it == args.end() ? fallback : it->second;
}

int int_arg_or(const std::map<std::string, std::string> &args,
               const std::string &key, int fallback) {
  const auto it = args.find(key);
  return it == args.end() ? fallback : std::stoi(it->second);
}

std::size_t mib_arg_or(const std::map<std::string, std::string> &args,
                       const std::string &key, std::size_t fallback_mib) {
  const auto it = args.find(key);
  const std::size_t mib =
      it == args.end() ? fallback_mib
                       : static_cast<std::size_t>(std::stoull(it->second));
  return mib * 1024ULL * 1024ULL;
}

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

double bytes_to_mib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

RunResult run_shape(nvinfer1::ICudaEngine &engine,
                    nvinfer1::IExecutionContext &context,
                    const std::string &input_name,
                    const std::string &output_name, int profile_index,
                    int batch, int width, cudaStream_t stream) {
  if (!context.setOptimizationProfileAsync(profile_index, stream)) {
    throw std::runtime_error("setOptimizationProfileAsync failed for profile " +
                             std::to_string(profile_index));
  }
  const nvinfer1::Dims4 input_dims{batch, 3, 48, width};
  if (!context.setInputShape(input_name.c_str(), input_dims)) {
    throw std::runtime_error("setInputShape failed for profile " +
                             std::to_string(profile_index) + " dims=" +
                             ppocrv6_native::engine::dims_to_string(input_dims));
  }
  const auto output_dims = context.getTensorShape(output_name.c_str());
  if (output_dims.nbDims != 3 || output_dims.d[0] != batch ||
      output_dims.d[2] != 192) {
    throw std::runtime_error("unexpected output dims for profile " +
                             std::to_string(profile_index) + ": " +
                             ppocrv6_native::engine::dims_to_string(
                                 output_dims));
  }

  const auto input_count =
      static_cast<std::size_t>(batch) * 3U * 48U *
      static_cast<std::size_t>(width);
  const auto output_count = volume(output_dims);
  ppocrv6_native::CudaPtr<float> input(input_count);
  ppocrv6_native::CudaPtr<std::uint16_t> output(output_count);
  PPOCRV6_CUDA_CHECK(cudaMemsetAsync(input.get(), 0,
                                     input_count * sizeof(float), stream));
  PPOCRV6_CUDA_CHECK(cudaMemsetAsync(output.get(), 0,
                                     output_count * sizeof(std::uint16_t),
                                     stream));

  if (!context.setTensorAddress(input_name.c_str(), input.get()) ||
      !context.setTensorAddress(output_name.c_str(), output.get())) {
    throw std::runtime_error("setTensorAddress failed");
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  if (!context.enqueueV3(stream)) {
    throw std::runtime_error("enqueueV3 failed for profile " +
                             std::to_string(profile_index));
  }
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));

  // Touch the output so CUDA cannot elide the allocation path in future edits.
  std::vector<std::uint16_t> host_sample(std::min<std::size_t>(output_count, 16));
  if (!host_sample.empty()) {
    PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(
        host_sample.data(), output.get(),
        host_sample.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
        stream));
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return RunResult{
      profile_index,
      batch,
      width,
      output_dims,
      elapsed_ms,
      input_count * sizeof(float),
      output_count * sizeof(std::uint16_t),
  };
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);
  const std::string onnx_path = arg_or(
      args, "--onnx", "artifacts/ppocrv6-medium/derived/rec-hidden.onnx");
  const std::string engine_path = arg_or(
      args, "--engine",
      "artifacts/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt");
  const bool force_rebuild = int_arg_or(args, "--force-rebuild", 0) != 0;
  const std::size_t workspace_bytes =
      mib_arg_or(args, "--workspace-mib", 512);
  const int opt_level = int_arg_or(args, "--opt-level", 3);

  ppocrv6_native::engine::RecognitionProfile glyph;
  glyph.min_batch = int_arg_or(args, "--glyph-min-batch", 1);
  glyph.opt_batch = int_arg_or(args, "--glyph-opt-batch", 128);
  glyph.max_batch = int_arg_or(args, "--glyph-max-batch", 256);
  glyph.min_width = int_arg_or(args, "--glyph-min-width", 48);
  glyph.opt_width = int_arg_or(args, "--glyph-opt-width", 80);
  glyph.max_width = int_arg_or(args, "--glyph-max-width", 128);
  glyph.workspace_bytes = workspace_bytes;
  glyph.builder_optimization_level = opt_level;

  ppocrv6_native::engine::RecognitionProfile short_line;
  short_line.min_batch = int_arg_or(args, "--short-line-min-batch", 1);
  short_line.opt_batch = int_arg_or(args, "--short-line-opt-batch", 8);
  short_line.max_batch = int_arg_or(args, "--short-line-max-batch", 12);
  short_line.min_width = int_arg_or(args, "--short-line-min-width", 128);
  short_line.opt_width = int_arg_or(args, "--short-line-opt-width", 384);
  short_line.max_width = int_arg_or(args, "--short-line-max-width", 640);
  short_line.workspace_bytes = workspace_bytes;
  short_line.builder_optimization_level = opt_level;

  ppocrv6_native::engine::RecognitionProfile line;
  line.min_batch = int_arg_or(args, "--line-min-batch", 1);
  line.opt_batch = int_arg_or(args, "--line-opt-batch", 8);
  line.max_batch = int_arg_or(args, "--line-max-batch", 12);
  line.min_width = int_arg_or(args, "--line-min-width", 640);
  line.opt_width = int_arg_or(args, "--line-opt-width", 1600);
  line.max_width = int_arg_or(args, "--line-max-width", 3200);
  line.workspace_bytes = workspace_bytes;
  line.builder_optimization_level = opt_level;

  if (force_rebuild || !std::filesystem::exists(engine_path)) {
    if (!ppocrv6_native::engine::build_recognition_hidden_engine(
            onnx_path, engine_path, std::vector{glyph, short_line, line})) {
      return 2;
    }
  }

  Logger logger;
  const auto engine_bytes = read_file(engine_path);
  auto runtime =
      std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("failed to create TensorRT runtime");
  }
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
  if (!engine) {
    throw std::runtime_error("failed to deserialize TensorRT engine");
  }

  std::string input_name;
  std::string output_name;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char *name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_name = name;
    } else {
      output_name = name;
    }
  }
  if (input_name.empty() || output_name.empty()) {
    throw std::runtime_error("failed to discover engine IO tensors");
  }

  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext());
  if (!context) {
    throw std::runtime_error("failed to create TensorRT execution context");
  }

  cudaStream_t stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream));

  const int glyph_batch = int_arg_or(args, "--inspect-glyph-batch", 128);
  const int glyph_width = int_arg_or(args, "--inspect-glyph-width", 80);
  const int short_line_batch =
      int_arg_or(args, "--inspect-short-line-batch", 8);
  const int short_line_width =
      int_arg_or(args, "--inspect-short-line-width", 384);
  const int line_batch = int_arg_or(args, "--inspect-line-batch", 4);
  const int line_width = int_arg_or(args, "--inspect-line-width", 3200);

  const auto glyph_result = run_shape(*engine, *context, input_name, output_name,
                                      0, glyph_batch, glyph_width, stream);
  const auto short_line_result =
      run_shape(*engine, *context, input_name, output_name, 1,
                short_line_batch, short_line_width, stream);
  const auto line_result = run_shape(*engine, *context, input_name, output_name,
                                     2, line_batch, line_width, stream);
  const auto glyph_after_line_result =
      run_shape(*engine, *context, input_name, output_name, 0, glyph_batch,
                glyph_width, stream);
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "trt_multi_profile_smoke: ok\n";
  std::cout << "engine=" << engine_path << '\n';
  std::cout << "engine_mib=" << bytes_to_mib(engine_bytes.size()) << '\n';
  std::cout << "profiles=" << engine->getNbOptimizationProfiles() << '\n';
  std::cout << "input=" << input_name << '\n';
  std::cout << "output=" << output_name << '\n';
  std::cout << "context_memory_all_profiles_mib="
            << bytes_to_mib(static_cast<std::size_t>(
                   std::max<int64_t>(engine->getDeviceMemorySizeV2(), 0)))
            << '\n';
  for (int profile = 0; profile < engine->getNbOptimizationProfiles();
       ++profile) {
    std::cout << "profile[" << profile << "].min="
              << ppocrv6_native::engine::dims_to_string(engine->getProfileShape(
                     input_name.c_str(), profile,
                     nvinfer1::OptProfileSelector::kMIN))
              << '\n';
    std::cout << "profile[" << profile << "].opt="
              << ppocrv6_native::engine::dims_to_string(engine->getProfileShape(
                     input_name.c_str(), profile,
                     nvinfer1::OptProfileSelector::kOPT))
              << '\n';
    std::cout << "profile[" << profile << "].max="
              << ppocrv6_native::engine::dims_to_string(engine->getProfileShape(
                     input_name.c_str(), profile,
                     nvinfer1::OptProfileSelector::kMAX))
              << '\n';
    const auto memory = engine->getDeviceMemorySizeForProfileV2(profile);
    std::cout << "profile[" << profile << "].context_memory_mib="
              << bytes_to_mib(static_cast<std::size_t>(
                     std::max<int64_t>(memory, 0)))
              << '\n';
  }

  const std::vector<RunResult> results{
      glyph_result,
      short_line_result,
      line_result,
      glyph_after_line_result,
  };
  for (const auto &result : results) {
    std::cout << "run.profile=" << result.profile << '\n';
    std::cout << "run.batch=" << result.batch << '\n';
    std::cout << "run.width=" << result.width << '\n';
    std::cout << "run.output_shape="
              << ppocrv6_native::engine::dims_to_string(result.output_dims)
              << '\n';
    std::cout << "run.enqueue_ms=" << result.enqueue_ms << '\n';
    std::cout << "run.input_mib=" << bytes_to_mib(result.input_bytes) << '\n';
    std::cout << "run.output_mib=" << bytes_to_mib(result.output_bytes) << '\n';
  }

  return 0;
}
