#include "ppocrv6_native/engine/onnx_to_trt.h"
#include "ppocrv6_native/engine/trt_engine.h"

#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"

#include <NvInfer.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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

struct MemorySnapshot {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

struct RunTiming {
  float h2d_ms = 0.0f;
  float execute_ms = 0.0f;
  float d2h_ms = 0.0f;
  float total_ms = 0.0f;
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

bool bool_arg_or(const std::map<std::string, std::string> &args,
                 const std::string &key, bool fallback) {
  const auto it = args.find(key);
  if (it == args.end()) {
    return fallback;
  }
  return it->second == "1" || it->second == "true" || it->second == "yes";
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

std::vector<float> read_binary_float(const std::string &path,
                                     std::size_t expected_count) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("failed to open " + path);
  }
  const std::streamsize bytes = in.tellg();
  const std::size_t expected_bytes = expected_count * sizeof(float);
  if (bytes < 0 || static_cast<std::size_t>(bytes) != expected_bytes) {
    throw std::runtime_error("unexpected input file size for " + path +
                             ": got " + std::to_string(bytes) +
                             " bytes, expected " +
                             std::to_string(expected_bytes));
  }
  std::vector<float> out(expected_count);
  in.seekg(0, std::ios::beg);
  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(expected_bytes));
  if (!in) {
    throw std::runtime_error("failed to read " + path);
  }
  return out;
}

void write_binary_bytes(const std::string &path,
                        const std::vector<unsigned char> &values) {
  const auto out_path = std::filesystem::path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output file: " + path);
  }
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size()));
  if (!out) {
    throw std::runtime_error("failed to write output file: " + path);
  }
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

std::size_t dtype_size(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
  case nvinfer1::DataType::kBOOL:
  case nvinfer1::DataType::kUINT8:
  case nvinfer1::DataType::kFP8:
    return 1;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kINT64:
    return 8;
  case nvinfer1::DataType::kBF16:
    return 2;
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
  case nvinfer1::DataType::kINT8:
    return "int8";
  case nvinfer1::DataType::kINT32:
    return "int32";
  case nvinfer1::DataType::kBOOL:
    return "bool";
  case nvinfer1::DataType::kUINT8:
    return "uint8";
  case nvinfer1::DataType::kFP8:
    return "fp8";
  case nvinfer1::DataType::kBF16:
    return "bf16";
  case nvinfer1::DataType::kINT64:
    return "int64";
  case nvinfer1::DataType::kINT4:
    return "int4";
  case nvinfer1::DataType::kFP4:
    return "fp4";
  }
  return "unknown";
}

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t stop) {
  PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  return elapsed;
}

double elapsed_wall_ms(std::chrono::steady_clock::time_point start,
                       std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

MemorySnapshot cuda_memory_snapshot() {
  MemorySnapshot snapshot;
  PPOCRV6_CUDA_CHECK(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
  return snapshot;
}

std::size_t used_bytes(const MemorySnapshot &snapshot) {
  return snapshot.total_bytes - snapshot.free_bytes;
}

double bytes_to_mib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_memory_snapshot(const std::string &label,
                           const MemorySnapshot &snapshot) {
  std::cout << "cuda_memory_" << label << "_free_mib="
            << bytes_to_mib(snapshot.free_bytes) << '\n';
  std::cout << "cuda_memory_" << label << "_used_mib="
            << bytes_to_mib(used_bytes(snapshot)) << '\n';
  std::cout << "cuda_memory_" << label << "_total_mib="
            << bytes_to_mib(snapshot.total_bytes) << '\n';
}

float percentile(std::vector<float> values, float q) {
  if (values.empty()) {
    return 0.0f;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::round(static_cast<float>(values.size() - 1) * q));
  return values[std::min(index, values.size() - 1)];
}

float mean(const std::vector<float> &values) {
  if (values.empty()) {
    return 0.0f;
  }
  return std::accumulate(values.begin(), values.end(), 0.0f) /
         static_cast<float>(values.size());
}

void print_distribution(const std::string &prefix,
                        const std::vector<float> &values) {
  std::cout << prefix << "_avg_ms=" << mean(values) << '\n';
  std::cout << prefix << "_min_ms=" << percentile(values, 0.0f) << '\n';
  std::cout << prefix << "_median_ms=" << percentile(values, 0.5f) << '\n';
  std::cout << prefix << "_p95_ms=" << percentile(values, 0.95f) << '\n';
}

std::string discover_input_name(nvinfer1::ICudaEngine &engine) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char *name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      return name;
    }
  }
  throw std::runtime_error("failed to discover engine input name");
}

std::string discover_output_name(nvinfer1::ICudaEngine &engine) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char *name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      return name;
    }
  }
  throw std::runtime_error("failed to discover engine output name");
}

RunTiming run_detector(nvinfer1::IExecutionContext &context,
                       const std::string &input_name,
                       const std::string &output_name, float *device_input,
                       const std::vector<float> &host_input,
                       unsigned char *device_output,
                       std::vector<unsigned char> &host_output,
                       cudaStream_t stream) {
  RunTiming timing;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));

  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(device_input, host_input.data(),
                                     host_input.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  timing.h2d_ms = elapsed_event_ms(start, stop);

  if (!context.setTensorAddress(input_name.c_str(), device_input) ||
      !context.setTensorAddress(output_name.c_str(), device_output)) {
    throw std::runtime_error("failed to set TensorRT tensor addresses");
  }

  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  if (!context.enqueueV3(stream)) {
    throw std::runtime_error("TensorRT detector enqueue failed");
  }
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  timing.execute_ms = elapsed_event_ms(start, stop);

  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(host_output.data(), device_output,
                                     host_output.size(),
                                     cudaMemcpyDeviceToHost, stream));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  timing.d2h_ms = elapsed_event_ms(start, stop);

  timing.total_ms = timing.h2d_ms + timing.execute_ms + timing.d2h_ms;
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
  return timing;
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);
  const std::string onnx_path = arg_or(
      args, "--onnx", "artifacts/ppocrv6-medium/source/det/inference.onnx");
  const std::string engine_path =
      arg_or(args, "--engine", "artifacts/ppocrv6-medium/engines/det-fp32-b1-h1280-w1280.trt");

  ppocrv6_native::engine::DetectionProfile profile;
  profile.min_batch = int_arg_or(args, "--min-batch", profile.min_batch);
  profile.opt_batch = int_arg_or(args, "--opt-batch", profile.opt_batch);
  profile.max_batch = int_arg_or(args, "--max-batch", profile.max_batch);
  profile.min_height = int_arg_or(args, "--min-height", profile.min_height);
  profile.opt_height = int_arg_or(args, "--opt-height", profile.opt_height);
  profile.max_height = int_arg_or(args, "--max-height", profile.max_height);
  profile.min_width = int_arg_or(args, "--min-width", profile.min_width);
  profile.opt_width = int_arg_or(args, "--opt-width", profile.opt_width);
  profile.max_width = int_arg_or(args, "--max-width", profile.max_width);
  profile.builder_optimization_level =
      int_arg_or(args, "--opt-level", profile.builder_optimization_level);
  profile.workspace_bytes = mib_arg_or(args, "--workspace-mib", 512);
  profile.fp16 = bool_arg_or(args, "--fp16", profile.fp16);

  const int inspect_batch =
      int_arg_or(args, "--inspect-batch", profile.opt_batch);
  const int inspect_height =
      int_arg_or(args, "--inspect-height", profile.opt_height);
  const int inspect_width =
      int_arg_or(args, "--inspect-width", profile.opt_width);
  const int warmups = int_arg_or(args, "--warmups", 2);
  const int repeats = int_arg_or(args, "--repeats", 5);
  const std::size_t max_extra_bytes =
      mib_arg_or(args, "--max-extra-mib", 1400);
  if (warmups < 0 || repeats <= 0 || inspect_batch <= 0 ||
      inspect_height <= 0 || inspect_width <= 0) {
    throw std::runtime_error("invalid detection benchmark arguments");
  }

  const bool force_rebuild = bool_arg_or(args, "--force-rebuild", false);
  if (force_rebuild || !std::filesystem::exists(engine_path)) {
    if (!ppocrv6_native::engine::build_detection_engine(onnx_path, engine_path,
                                                        profile)) {
      return 2;
    }
  }

  const auto init_total_start = std::chrono::steady_clock::now();

  const auto cuda_context_start = std::chrono::steady_clock::now();
  PPOCRV6_CUDA_CHECK(cudaFree(nullptr));
  const double init_cuda_context_ms =
      elapsed_wall_ms(cuda_context_start, std::chrono::steady_clock::now());
  const auto memory_after_cuda_context = cuda_memory_snapshot();

  const auto runtime_start = std::chrono::steady_clock::now();
  ppocrv6_native::engine::Logger logger;
  auto runtime =
      std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("failed to create TensorRT runtime");
  }
  const double init_runtime_ms =
      elapsed_wall_ms(runtime_start, std::chrono::steady_clock::now());
  const auto memory_after_runtime = cuda_memory_snapshot();

  const auto engine_load_start = std::chrono::steady_clock::now();
  const auto engine_bytes = read_file(engine_path);
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
  if (!engine) {
    throw std::runtime_error("failed to deserialize engine " + engine_path);
  }
  const double init_engine_load_ms =
      elapsed_wall_ms(engine_load_start, std::chrono::steady_clock::now());
  const auto memory_after_engine_load = cuda_memory_snapshot();

  const std::string input_name = discover_input_name(*engine);
  const std::string output_name = discover_output_name(*engine);
  const auto input_dtype = engine->getTensorDataType(input_name.c_str());
  const auto output_dtype = engine->getTensorDataType(output_name.c_str());
  if (input_dtype != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error("detector input dtype is not float32");
  }

  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!context) {
    throw std::runtime_error("failed to create detector execution context");
  }
  const std::size_t context_memory_bytes =
      static_cast<std::size_t>(engine->getDeviceMemorySizeV2());
  ppocrv6_native::CudaPtr<unsigned char> context_memory(context_memory_bytes);
  context->setDeviceMemoryV2(context_memory.get(),
                             static_cast<int64_t>(context_memory_bytes));

  cudaStream_t stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream));
  if (!context->setOptimizationProfileAsync(0, stream)) {
    throw std::runtime_error("failed to set detector optimization profile");
  }

  const nvinfer1::Dims4 input_dims{inspect_batch, 3, inspect_height,
                                   inspect_width};
  if (!context->setInputShape(input_name.c_str(), input_dims)) {
    throw std::runtime_error("failed to set detector input shape " +
                             ppocrv6_native::engine::dims_to_string(input_dims));
  }
  const auto output_dims = context->getTensorShape(output_name.c_str());
  const std::size_t input_count = volume(input_dims);
  const std::size_t output_count = volume(output_dims);
  if (input_count == 0 || output_count == 0) {
    throw std::runtime_error("invalid detector input/output volume");
  }
  const std::size_t input_bytes = input_count * sizeof(float);
  const std::size_t output_bytes = output_count * dtype_size(output_dtype);

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
  ppocrv6_native::CudaPtr<unsigned char> device_output(output_bytes);
  std::vector<unsigned char> host_output(output_bytes);

  const auto memory_after_all_alloc = cuda_memory_snapshot();
  const std::size_t extra_bytes =
      used_bytes(memory_after_all_alloc) - used_bytes(memory_after_cuda_context);
  if (extra_bytes > max_extra_bytes) {
    std::cerr << "extra_device_memory_mib=" << bytes_to_mib(extra_bytes)
              << " exceeds max_extra_mib=" << bytes_to_mib(max_extra_bytes)
              << '\n';
    PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));
    return 3;
  }

  const auto warmup_start = std::chrono::steady_clock::now();
  for (int i = 0; i < warmups; ++i) {
    (void)run_detector(*context, input_name, output_name, device_input.get(),
                       host_input, device_output.get(), host_output, stream);
  }
  const double init_warmup_ms =
      elapsed_wall_ms(warmup_start, std::chrono::steady_clock::now());
  const auto memory_after_warmup = cuda_memory_snapshot();
  const double init_total_before_repeats_ms =
      elapsed_wall_ms(init_total_start, std::chrono::steady_clock::now());

  std::vector<float> h2d_ms;
  std::vector<float> execute_ms;
  std::vector<float> d2h_ms;
  std::vector<float> total_ms;
  h2d_ms.reserve(static_cast<std::size_t>(repeats));
  execute_ms.reserve(static_cast<std::size_t>(repeats));
  d2h_ms.reserve(static_cast<std::size_t>(repeats));
  total_ms.reserve(static_cast<std::size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    const auto timing =
        run_detector(*context, input_name, output_name, device_input.get(),
                     host_input, device_output.get(), host_output, stream);
    h2d_ms.push_back(timing.h2d_ms);
    execute_ms.push_back(timing.execute_ms);
    d2h_ms.push_back(timing.d2h_ms);
    total_ms.push_back(timing.total_ms);
  }
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));

  const auto output_it = args.find("--output");
  if (output_it != args.end()) {
    write_binary_bytes(output_it->second, host_output);
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "trt_detection_bench: ok\n";
  std::cout << "onnx=" << onnx_path << '\n';
  std::cout << "engine=" << engine_path << '\n';
  std::cout << "profiles=" << engine->getNbOptimizationProfiles() << '\n';
  std::cout << "input_name=" << input_name << '\n';
  std::cout << "output_name=" << output_name << '\n';
  std::cout << "input_dtype=" << dtype_name(input_dtype) << '\n';
  std::cout << "output_dtype=" << dtype_name(output_dtype) << '\n';
  std::cout << "inspect_input_shape="
            << ppocrv6_native::engine::dims_to_string(input_dims) << '\n';
  std::cout << "inspect_output_shape="
            << ppocrv6_native::engine::dims_to_string(output_dims) << '\n';
  std::cout << "warmups=" << warmups << '\n';
  std::cout << "repeats=" << repeats << '\n';
  std::cout << "fp16=" << (profile.fp16 ? 1 : 0) << '\n';
  std::cout << "init_cuda_context_ms=" << init_cuda_context_ms << '\n';
  std::cout << "init_runtime_ms=" << init_runtime_ms << '\n';
  std::cout << "init_engine_load_ms=" << init_engine_load_ms << '\n';
  std::cout << "init_warmup_ms=" << init_warmup_ms << '\n';
  std::cout << "init_total_before_repeats_ms="
            << init_total_before_repeats_ms << '\n';
  std::cout << "engine_mib=" << bytes_to_mib(engine_bytes.size()) << '\n';
  std::cout << "context_memory_mib="
            << bytes_to_mib(context_memory_bytes) << '\n';
  std::cout << "input_mib=" << bytes_to_mib(input_bytes) << '\n';
  std::cout << "output_mib=" << bytes_to_mib(output_bytes) << '\n';
  std::cout << "explicit_total_mib="
            << bytes_to_mib(context_memory_bytes + input_bytes + output_bytes)
            << '\n';
  std::cout << "extra_device_memory_since_cuda_context_mib="
            << bytes_to_mib(extra_bytes) << '\n';
  print_memory_snapshot("after_cuda_context", memory_after_cuda_context);
  print_memory_snapshot("after_runtime", memory_after_runtime);
  print_memory_snapshot("after_engine_load", memory_after_engine_load);
  print_memory_snapshot("after_all_alloc", memory_after_all_alloc);
  print_memory_snapshot("after_warmup", memory_after_warmup);
  print_distribution("h2d", h2d_ms);
  print_distribution("execute", execute_ms);
  print_distribution("d2h", d2h_ms);
  print_distribution("total", total_ms);
  if (output_it != args.end()) {
    std::cout << "output_path=" << output_it->second << '\n';
  }
  return 0;
}
