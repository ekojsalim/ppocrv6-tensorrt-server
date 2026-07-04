#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/engine/trt_engine.h"
#include "ppocrv6_native/kernels/classifier.h"
#include "ppocrv6_native/plugin/classifier_plugin.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

struct BuildLogger final : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT plugin] " << msg << '\n';
    }
  }
};

struct PluginEngineBundle {
  // The in-memory V2 plugin engine can still reference build-time state while
  // contexts are created, so keep these objects alive for the smoke run.
  std::unique_ptr<BuildLogger> logger;
  std::unique_ptr<nvinfer1::IBuilder> builder;
  std::unique_ptr<ppocrv6_native::plugin::WmmaClassifierPlugin> plugin;
  std::unique_ptr<nvinfer1::INetworkDefinition> network;
  std::unique_ptr<nvinfer1::IBuilderConfig> config;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
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

const std::string &required(const std::map<std::string, std::string> &args,
                            const std::string &key) {
  const auto it = args.find(key);
  if (it == args.end()) {
    throw std::runtime_error("missing required argument " + key);
  }
  return it->second;
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

template <typename T>
void write_binary(const std::string &path, const std::vector<T> &values) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output " + path);
  }
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out) {
    throw std::runtime_error("failed to write output " + path);
  }
}

double bytes_to_mib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

struct MemorySnapshot {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

MemorySnapshot cuda_memory_snapshot() {
  MemorySnapshot snapshot;
  PPOCRV6_CUDA_CHECK(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
  return snapshot;
}

void print_memory_snapshot(const std::string &label,
                           const MemorySnapshot &snapshot) {
  const std::size_t used_bytes = snapshot.total_bytes - snapshot.free_bytes;
  std::cout << "cuda_memory_" << label << "_free_mib="
            << bytes_to_mib(snapshot.free_bytes) << '\n';
  std::cout << "cuda_memory_" << label << "_used_mib="
            << bytes_to_mib(used_bytes) << '\n';
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

PluginEngineBundle build_plugin_engine(
    int batch, int timesteps, int hidden_size, int vocab_size,
    int vocab_tile_size, std::size_t workspace_bytes, bool trace) {
  const auto trace_step = [trace](const char *message) {
    if (trace) {
      std::cerr << "[plugin-smoke] " << message << '\n';
    }
  };

  trace_step("create builder");
  PluginEngineBundle bundle;
  bundle.logger = std::make_unique<BuildLogger>();
  bundle.builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(*bundle.logger));
  if (!bundle.builder) {
    throw std::runtime_error("failed to create TensorRT builder");
  }
  trace_step("create network");
  bundle.network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      bundle.builder->createNetworkV2(0U));
  if (!bundle.network) {
    throw std::runtime_error("failed to create TensorRT network");
  }

  trace_step("add inputs");
  auto *hidden = bundle.network->addInput(
      "hidden", nvinfer1::DataType::kHALF,
      nvinfer1::Dims3{-1, -1, hidden_size});
  auto *weight = bundle.network->addInput(
      "weight", nvinfer1::DataType::kHALF,
      nvinfer1::Dims2{hidden_size, vocab_size});
  nvinfer1::Dims bias_dims{};
  bias_dims.nbDims = 1;
  bias_dims.d[0] = vocab_size;
  auto *bias =
      bundle.network->addInput("bias", nvinfer1::DataType::kHALF, bias_dims);
  if (hidden == nullptr || weight == nullptr || bias == nullptr) {
    throw std::runtime_error("failed to add plugin network inputs");
  }

  trace_step("add plugin");
  bundle.plugin =
      std::make_unique<ppocrv6_native::plugin::WmmaClassifierPlugin>(
          vocab_tile_size);
  nvinfer1::ITensor *plugin_inputs[] = {hidden, weight, bias};
  auto *layer = bundle.network->addPluginV2(plugin_inputs, 3, *bundle.plugin);
  if (layer == nullptr || layer->getNbOutputs() != 3) {
    throw std::runtime_error("failed to add WMMA classifier plugin layer");
  }
  layer->getOutput(0)->setName("indices");
  layer->getOutput(1)->setName("prob");
  layer->getOutput(2)->setName("max_logits");
  bundle.network->markOutput(*layer->getOutput(0));
  bundle.network->markOutput(*layer->getOutput(1));
  bundle.network->markOutput(*layer->getOutput(2));

  trace_step("create config");
  bundle.config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      bundle.builder->createBuilderConfig());
  if (!bundle.config) {
    throw std::runtime_error("failed to create TensorRT builder config");
  }
  bundle.config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                    workspace_bytes);
  if (bundle.builder->platformHasFastFp16()) {
    bundle.config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  trace_step("create profile");
  auto *profile = bundle.builder->createOptimizationProfile();
  if (profile == nullptr) {
    throw std::runtime_error("failed to create TensorRT optimization profile");
  }
  profile->setDimensions("hidden", nvinfer1::OptProfileSelector::kMIN,
                         nvinfer1::Dims3{batch, timesteps, hidden_size});
  profile->setDimensions("hidden", nvinfer1::OptProfileSelector::kOPT,
                         nvinfer1::Dims3{batch, timesteps, hidden_size});
  profile->setDimensions("hidden", nvinfer1::OptProfileSelector::kMAX,
                         nvinfer1::Dims3{batch, timesteps, hidden_size});
  if (!profile->isValid()) {
    throw std::runtime_error("TensorRT plugin profile is invalid");
  }
  bundle.config->addOptimizationProfile(profile);

  trace_step("build engine");
  auto *engine =
      bundle.builder->buildEngineWithConfig(*bundle.network, *bundle.config);
  if (engine == nullptr) {
    throw std::runtime_error("TensorRT plugin engine build failed");
  }
  trace_step("engine built");
  bundle.engine.reset(engine);
  return bundle;
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);
  const int batch = int_arg_or(args, "--batch", 1);
  const int timesteps = int_arg_or(args, "--timesteps", 80);
  const int hidden_size = int_arg_or(args, "--hidden-size", 192);
  const int vocab_size = int_arg_or(args, "--vocab-size", 18710);
  const int vocab_tile_size = int_arg_or(args, "--vocab-tile-size", 512);
  const int warmups = int_arg_or(args, "--warmups", 10);
  const int repeats = int_arg_or(args, "--repeats", 50);
  const bool trace = int_arg_or(args, "--trace", 0) != 0;
  const std::size_t workspace_bytes =
      mib_arg_or(args, "--workspace-mib", 128);
  if (batch <= 0 || timesteps <= 0 || hidden_size <= 0 || vocab_size <= 0 ||
      vocab_tile_size <= 0 || warmups < 0 || repeats <= 0) {
    throw std::runtime_error("invalid plugin smoke arguments");
  }

  const int rows = batch * timesteps;
  const std::size_t hidden_count =
      static_cast<std::size_t>(rows) * hidden_size;
  const std::size_t weight_count =
      static_cast<std::size_t>(hidden_size) * vocab_size;
  const std::size_t bias_count = static_cast<std::size_t>(vocab_size);

  const auto hidden =
      read_binary<std::uint16_t>(required(args, "--hidden"), hidden_count);
  const auto weight =
      read_binary<std::uint16_t>(required(args, "--weight"), weight_count);
  const auto bias =
      read_binary<std::uint16_t>(required(args, "--bias"), bias_count);

  PPOCRV6_CUDA_CHECK(cudaFree(nullptr));
  if (trace) {
    std::cerr << "[plugin-smoke] build engine\n";
  }
  auto plugin_engine = build_plugin_engine(batch, timesteps, hidden_size,
                                           vocab_size, vocab_tile_size,
                                           workspace_bytes, trace);
  const auto memory_after_engine = cuda_memory_snapshot();
  if (trace) {
    std::cerr << "[plugin-smoke] create context\n";
  }
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
      plugin_engine.engine->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!context) {
    throw std::runtime_error("failed to create TensorRT plugin execution context");
  }
  const auto context_memory_bytes = plugin_engine.engine->getDeviceMemorySizeV2();
  if (trace) {
    std::cerr << "[plugin-smoke] context memory bytes="
              << context_memory_bytes << '\n';
  }
  ppocrv6_native::CudaPtr<std::uint8_t> d_context_memory(
      context_memory_bytes > 0
          ? static_cast<std::size_t>(context_memory_bytes)
          : 0);
  context->setDeviceMemoryV2(d_context_memory.get(), context_memory_bytes);
  const auto memory_after_context = cuda_memory_snapshot();
  if (trace) {
    std::cerr << "[plugin-smoke] set input shape\n";
  }
  if (!context->setInputShape("hidden",
                              nvinfer1::Dims3{batch, timesteps, hidden_size})) {
    throw std::runtime_error("failed to set plugin hidden input shape");
  }
  const auto indices_shape = context->getTensorShape("indices");
  const auto prob_shape = context->getTensorShape("prob");
  const auto max_shape = context->getTensorShape("max_logits");
  if (indices_shape.nbDims != 2 || indices_shape.d[0] != batch ||
      indices_shape.d[1] != timesteps || prob_shape.nbDims != 2 ||
      max_shape.nbDims != 2) {
    throw std::runtime_error("unexpected plugin output shapes");
  }

  if (trace) {
    std::cerr << "[plugin-smoke] allocate device buffers\n";
  }
  ppocrv6_native::CudaPtr<std::uint16_t> d_hidden(hidden_count);
  ppocrv6_native::CudaPtr<std::uint16_t> d_weight(weight_count);
  ppocrv6_native::CudaPtr<std::uint16_t> d_bias(bias_count);
  ppocrv6_native::CudaPtr<int> d_indices(static_cast<std::size_t>(rows));
  ppocrv6_native::CudaPtr<float> d_prob(static_cast<std::size_t>(rows));
  ppocrv6_native::CudaPtr<float> d_max_logits(static_cast<std::size_t>(rows));
  const auto memory_after_alloc = cuda_memory_snapshot();
  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_hidden.get(), hidden.data(),
                                hidden.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_weight.get(), weight.data(),
                                weight.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(d_bias.get(), bias.data(),
                                bias.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice));

  if (trace) {
    std::cerr << "[plugin-smoke] bind tensor addresses\n";
  }
  context->setTensorAddress("hidden", d_hidden.get());
  context->setTensorAddress("weight", d_weight.get());
  context->setTensorAddress("bias", d_bias.get());
  context->setTensorAddress("indices", d_indices.get());
  context->setTensorAddress("prob", d_prob.get());
  context->setTensorAddress("max_logits", d_max_logits.get());

  cudaStream_t stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream));
  if (trace) {
    std::cerr << "[plugin-smoke] warmups\n";
  }
  for (int i = 0; i < warmups; ++i) {
    if (!context->enqueueV3(stream)) {
      throw std::runtime_error("plugin warmup enqueue failed");
    }
  }
  PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto memory_after_warmup = cuda_memory_snapshot();

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> run_ms;
  run_ms.reserve(static_cast<std::size_t>(repeats));
  if (trace) {
    std::cerr << "[plugin-smoke] timed repeats\n";
  }
  for (int i = 0; i < repeats; ++i) {
    PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
    if (!context->enqueueV3(stream)) {
      throw std::runtime_error("plugin timed enqueue failed");
    }
    PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
    PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    run_ms.push_back(elapsed);
  }
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));

  std::vector<int> indices(static_cast<std::size_t>(rows));
  std::vector<float> prob(static_cast<std::size_t>(rows));
  std::vector<float> max_logits(static_cast<std::size_t>(rows));
  if (trace) {
    std::cerr << "[plugin-smoke] copy outputs\n";
  }
  PPOCRV6_CUDA_CHECK(cudaMemcpy(indices.data(), d_indices.get(),
                                indices.size() * sizeof(int),
                                cudaMemcpyDeviceToHost));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(prob.data(), d_prob.get(),
                                prob.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(max_logits.data(), d_max_logits.get(),
                                max_logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));

  const auto indices_out = args.find("--out-indices");
  if (indices_out != args.end()) {
    write_binary(indices_out->second, indices);
  }
  const auto prob_out = args.find("--out-prob");
  if (prob_out != args.end()) {
    write_binary(prob_out->second, prob);
  }
  const auto max_out = args.find("--out-max-logits");
  if (max_out != args.end()) {
    write_binary(max_out->second, max_logits);
  }

  const auto minmax_prob = std::minmax_element(prob.begin(), prob.end());
  const auto minmax_logits =
      std::minmax_element(max_logits.begin(), max_logits.end());
  const float total_ms = std::accumulate(run_ms.begin(), run_ms.end(), 0.0f);
  const auto workspace_shape =
      ppocrv6_native::kernels::tiled_classifier_workspace_shape(
          rows, vocab_size, 16, vocab_tile_size);
  const std::size_t partial_bytes =
      workspace_shape.partial_count *
      (sizeof(int) + sizeof(float) + sizeof(float));
  const std::size_t hidden_bytes = hidden_count * sizeof(std::uint16_t);
  const std::size_t weight_bytes = weight_count * sizeof(std::uint16_t);
  const std::size_t bias_bytes = bias_count * sizeof(std::uint16_t);
  const std::size_t compact_output_bytes =
      static_cast<std::size_t>(rows) *
      (sizeof(int) + sizeof(float) + sizeof(float));
  const std::size_t context_bytes =
      context_memory_bytes > 0 ? static_cast<std::size_t>(context_memory_bytes)
                               : 0;
  const std::size_t explicit_device_buffer_bytes =
      hidden_bytes + weight_bytes + bias_bytes + compact_output_bytes +
      context_bytes;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "trt_classifier_plugin_smoke: ok\n";
  std::cout << "shape=[" << batch << "," << timesteps << "," << hidden_size
            << "] vocab_size=" << vocab_size
            << " vocab_tile_size=" << vocab_tile_size << '\n';
  std::cout << "execute_repeats=" << repeats << '\n';
  std::cout << "execute_total_ms=" << total_ms << '\n';
  std::cout << "execute_avg_ms=" << mean(run_ms) << '\n';
  std::cout << "execute_min_ms=" << percentile(run_ms, 0.0f) << '\n';
  std::cout << "execute_median_ms=" << percentile(run_ms, 0.5f) << '\n';
  std::cout << "execute_p95_ms=" << percentile(run_ms, 0.95f) << '\n';
  std::cout << "partial_count=" << workspace_shape.partial_count << '\n';
  std::cout << "partial_bytes=" << partial_bytes << '\n';
  std::cout << "partial_mib=" << bytes_to_mib(partial_bytes) << '\n';
  std::cout << "trt_context_memory_bytes=" << context_bytes << '\n';
  std::cout << "trt_context_memory_mib=" << bytes_to_mib(context_bytes) << '\n';
  std::cout << "hidden_bytes=" << hidden_bytes << '\n';
  std::cout << "weight_bytes=" << weight_bytes << '\n';
  std::cout << "bias_bytes=" << bias_bytes << '\n';
  std::cout << "compact_output_bytes=" << compact_output_bytes << '\n';
  std::cout << "explicit_device_buffer_bytes="
            << explicit_device_buffer_bytes << '\n';
  std::cout << "explicit_device_buffer_mib="
            << bytes_to_mib(explicit_device_buffer_bytes) << '\n';
  print_memory_snapshot("after_engine", memory_after_engine);
  print_memory_snapshot("after_context", memory_after_context);
  print_memory_snapshot("after_alloc", memory_after_alloc);
  print_memory_snapshot("after_warmup", memory_after_warmup);
  std::cout << "prob_min=" << *minmax_prob.first << " prob_max="
            << *minmax_prob.second << '\n';
  std::cout << "max_logit_min=" << *minmax_logits.first
            << " max_logit_max=" << *minmax_logits.second << '\n';
  std::cout << "first_ids=";
  for (int i = 0; i < std::min(rows, 8); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << indices[static_cast<std::size_t>(i)];
  }
  std::cout << '\n';
  return 0;
}
