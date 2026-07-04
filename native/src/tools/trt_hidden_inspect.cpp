#include "ppocrv6_native/engine/onnx_to_trt.h"
#include "ppocrv6_native/engine/trt_engine.h"

#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"

#include <NvInfer.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
      it == args.end() ? fallback_mib : static_cast<std::size_t>(std::stoull(it->second));
  return mib * 1024ULL * 1024ULL;
}

bool has_flag(const std::map<std::string, std::string> &args,
              const std::string &key) {
  const auto it = args.find(key);
  return it != args.end() && (it->second == "1" || it->second == "true" ||
                              it->second == "yes");
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

void write_binary(const std::string &path, const std::vector<std::uint16_t> &values) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
  if (!out) {
    throw std::runtime_error("failed to write output file: " + path);
  }
}

std::vector<float> read_binary_float(const std::string &path,
                                     std::size_t expected_count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to open input file: " + path);
  }
  const std::streamsize bytes = in.tellg();
  const std::size_t expected_bytes = expected_count * sizeof(float);
  if (bytes < 0 || static_cast<std::size_t>(bytes) != expected_bytes) {
    throw std::runtime_error("unexpected input file size for " + path +
                             ": got " + std::to_string(bytes) +
                             " bytes, expected " +
                             std::to_string(expected_bytes));
  }
  std::vector<float> values(expected_count);
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(values.data()),
          static_cast<std::streamsize>(expected_bytes));
  if (!in) {
    throw std::runtime_error("failed to read input file: " + path);
  }
  return values;
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);
  const std::string onnx_path = arg_or(
      args, "--onnx", "artifacts/ppocrv6-medium/derived/rec-hidden.onnx");
  const std::string engine_path = arg_or(
      args, "--engine",
      "artifacts/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt");

  ppocrv6_native::engine::RecognitionProfile profile;
  profile.min_batch = int_arg_or(args, "--min-batch", profile.min_batch);
  profile.opt_batch = int_arg_or(args, "--opt-batch", profile.opt_batch);
  profile.max_batch = int_arg_or(args, "--max-batch", profile.max_batch);
  profile.min_width = int_arg_or(args, "--min-width", profile.min_width);
  profile.opt_width = int_arg_or(args, "--opt-width", profile.opt_width);
  profile.max_width = int_arg_or(args, "--max-width", profile.max_width);
  profile.builder_optimization_level =
      int_arg_or(args, "--opt-level", profile.builder_optimization_level);
  profile.workspace_bytes = mib_arg_or(args, "--workspace-mib", 1024);

  if (!std::filesystem::exists(engine_path)) {
    if (!ppocrv6_native::engine::build_recognition_hidden_engine(
            onnx_path, engine_path, profile)) {
      return 2;
    }
  }

  ppocrv6_native::engine::TrtEngine engine(engine_path);
  if (!engine.load()) {
    return 3;
  }

  std::cout << "engine=" << engine_path << '\n';
  std::cout << "profiles=" << engine.num_profiles() << '\n';
  for (const auto &name : engine.input_names()) {
    std::cout << "input=" << name << '\n';
  }
  for (const auto &name : engine.output_names()) {
    std::cout << "output=" << name << '\n';
  }

  if (engine.input_names().empty() || engine.output_names().empty()) {
    std::cerr << "engine has no input or output tensors\n";
    return 4;
  }

  const int inspect_batch = int_arg_or(args, "--inspect-batch", profile.opt_batch);
  const int inspect_width = int_arg_or(args, "--inspect-width", profile.opt_width);
  const nvinfer1::Dims4 input_dims{inspect_batch, 3, 48, inspect_width};
  if (!engine.set_input_shape(engine.input_names()[0], input_dims)) {
    return 5;
  }
  const auto output_dims = engine.tensor_shape(engine.output_names()[0]);
  std::cout << "inspect_input_shape="
            << ppocrv6_native::engine::dims_to_string(input_dims) << '\n';
  std::cout << "inspect_output_shape="
            << ppocrv6_native::engine::dims_to_string(output_dims) << '\n';
  if (output_dims.nbDims == 3) {
    std::cout << "actual_time_steps=" << output_dims.d[1] << '\n';
    std::cout << "floor_width_div8=" << inspect_width / 8 << '\n';
    std::cout << "width_mod8=" << inspect_width % 8 << '\n';
    std::cout << "hidden_size=" << output_dims.d[2] << '\n';
  }

  if (!has_flag(args, "--execute")) {
    return 0;
  }

  const std::size_t input_count = static_cast<std::size_t>(inspect_batch) * 3 *
                                  48 * static_cast<std::size_t>(inspect_width);
  const std::size_t output_count = volume(output_dims);
  if (input_count == 0 || output_count == 0) {
    std::cerr << "invalid input/output volume\n";
    return 6;
  }

  std::vector<float> host_input;
  const auto input_it = args.find("--input");
  if (input_it == args.end()) {
    host_input.resize(input_count);
    for (std::size_t i = 0; i < host_input.size(); ++i) {
      host_input[i] =
          static_cast<float>(static_cast<int>(i % 251) - 125) / 125.0f;
    }
  } else {
    host_input = read_binary_float(input_it->second, input_count);
  }

  ppocrv6_native::CudaPtr<float> device_input(input_count);
  ppocrv6_native::CudaPtr<std::uint16_t> device_output(output_count);
  PPOCRV6_CUDA_CHECK(cudaMemcpy(device_input.get(), host_input.data(),
                                host_input.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemset(device_output.get(), 0,
                                output_count * sizeof(std::uint16_t)));

  engine.set_tensor_address(engine.input_names()[0], device_input.get());
  engine.set_tensor_address(engine.output_names()[0], device_output.get());

  cudaStream_t stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream));

  const int warmups = int_arg_or(args, "--warmups", 2);
  const int repeats = int_arg_or(args, "--repeats", 5);
  for (int i = 0; i < warmups; ++i) {
    if (!engine.execute(stream)) {
      std::cerr << "TensorRT execute failed during warmup\n";
      return 7;
    }
  }
  PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    if (!engine.execute(stream)) {
      std::cerr << "TensorRT execute failed during timed run\n";
      return 8;
    }
  }
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));

  std::vector<std::uint16_t> host_output(output_count);
  PPOCRV6_CUDA_CHECK(cudaMemcpy(host_output.data(), device_output.get(),
                                host_output.size() * sizeof(std::uint16_t),
                                cudaMemcpyDeviceToHost));
  const std::size_t nonzero =
      static_cast<std::size_t>(std::count_if(host_output.begin(), host_output.end(),
                                             [](std::uint16_t v) { return v != 0; }));
  const auto output_it = args.find("--output");
  if (output_it != args.end()) {
    write_binary(output_it->second, host_output);
  }

  std::cout << "execute_repeats=" << repeats << '\n';
  std::cout << "execute_total_ms=" << elapsed_ms << '\n';
  std::cout << "execute_avg_ms=" << elapsed_ms / static_cast<float>(std::max(repeats, 1))
            << '\n';
  std::cout << "input_bytes=" << input_count * sizeof(float) << '\n';
  std::cout << "output_fp16_bytes=" << output_count * sizeof(std::uint16_t)
            << '\n';
  std::cout << "output_nonzero=" << nonzero << '\n';
  return 0;
}
