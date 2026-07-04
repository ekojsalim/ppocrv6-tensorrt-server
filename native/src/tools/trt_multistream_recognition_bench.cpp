#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/engine/trt_engine.h"
#include "ppocrv6_native/kernels/classifier.h"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Logger final : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT multistream] " << msg << '\n';
    }
  }
};

struct Args {
  std::map<std::string, std::string> values;
  std::vector<std::string> slots;
};

struct SlotSpec {
  std::string name;
  std::string engine_path;
  std::string input_path;
  int batch = 1;
  int width = 640;
};

struct MemorySnapshot {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

struct ClassifierBuffers {
  ppocrv6_native::CudaPtr<std::uint16_t> weight;
  ppocrv6_native::CudaPtr<std::uint16_t> bias;
  int vocab_size = 0;
  int hidden_size = 0;
  std::size_t weight_bytes = 0;
  std::size_t bias_bytes = 0;
};

struct SlotRuntime {
  SlotSpec spec;
  nvinfer1::ICudaEngine *engine = nullptr;
  ppocrv6_native::CudaPtr<std::uint8_t> context_memory;
  ppocrv6_native::CudaPtr<float> input;
  ppocrv6_native::CudaPtr<std::uint16_t> hidden;
  ppocrv6_native::CudaPtr<int> indices;
  ppocrv6_native::CudaPtr<float> prob;
  ppocrv6_native::CudaPtr<float> max_logits;
  ppocrv6_native::CudaPtr<int> partial_ids;
  ppocrv6_native::CudaPtr<float> partial_max;
  ppocrv6_native::CudaPtr<float> partial_sum;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::string input_name;
  std::string output_name;
  nvinfer1::Dims4 input_dims{};
  nvinfer1::Dims output_dims{};
  ppocrv6_native::kernels::TiledClassifierWorkspaceShape workspace;
  cudaStream_t stream = nullptr;
  cudaEvent_t stop_event = nullptr;
  int timesteps = 0;
  int hidden_size = 0;
  int rows = 0;
  std::size_t input_count = 0;
  std::size_t hidden_count = 0;
  std::size_t context_memory_bytes = 0;
  std::size_t input_bytes = 0;
  std::size_t hidden_bytes = 0;
  std::size_t compact_output_bytes = 0;
  std::size_t partial_bytes = 0;
  std::vector<int> sample_ids;
  float prob_min = 0.0f;
  float prob_max = 0.0f;

  SlotRuntime() = default;
  SlotRuntime(const SlotRuntime &) = delete;
  SlotRuntime &operator=(const SlotRuntime &) = delete;

  ~SlotRuntime() {
    if (stop_event != nullptr) {
      cudaEventDestroy(stop_event);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    context.reset();
  }

  [[nodiscard]] std::size_t explicit_bytes() const {
    return context_memory_bytes + input_bytes + hidden_bytes +
           compact_output_bytes + partial_bytes;
  }
};

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) {
      throw std::runtime_error("unexpected positional argument: " + key);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + key);
    }
    const std::string value = argv[++i];
    if (key == "--slot") {
      args.slots.push_back(value);
    } else {
      args.values[key] = value;
    }
  }
  return args;
}

std::string arg_or(const Args &args, const std::string &key,
                   const std::string &fallback) {
  const auto it = args.values.find(key);
  return it == args.values.end() ? fallback : it->second;
}

int int_arg_or(const Args &args, const std::string &key, int fallback) {
  const auto it = args.values.find(key);
  return it == args.values.end() ? fallback : std::stoi(it->second);
}

std::size_t mib_arg_or(const Args &args, const std::string &key,
                       std::size_t fallback_mib) {
  const auto it = args.values.find(key);
  const std::size_t mib =
      it == args.values.end() ? fallback_mib
                              : static_cast<std::size_t>(std::stoull(it->second));
  return mib * 1024ULL * 1024ULL;
}

std::vector<std::string> split_csv(const std::string &value) {
  std::vector<std::string> out;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    out.push_back(item);
  }
  return out;
}

SlotSpec parse_slot_spec(const std::string &value) {
  const auto parts = split_csv(value);
  if (parts.size() != 5) {
    throw std::runtime_error(
        "--slot must be name,engine_path,input_path,batch,width");
  }
  SlotSpec spec;
  spec.name = parts[0];
  spec.engine_path = parts[1];
  spec.input_path = parts[2];
  spec.batch = std::stoi(parts[3]);
  spec.width = std::stoi(parts[4]);
  if (spec.name.empty() || spec.engine_path.empty() || spec.input_path.empty() ||
      spec.batch <= 0 || spec.width <= 0) {
    throw std::runtime_error("invalid --slot value: " + value);
  }
  return spec;
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

MemorySnapshot cuda_memory_snapshot() {
  MemorySnapshot snapshot;
  PPOCRV6_CUDA_CHECK(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
  return snapshot;
}

std::size_t used_bytes(const MemorySnapshot &snapshot) {
  return snapshot.total_bytes - snapshot.free_bytes;
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

std::string io_mode_name(nvinfer1::TensorIOMode mode) {
  return mode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output";
}

std::unique_ptr<SlotRuntime> create_slot(
    nvinfer1::ICudaEngine &engine, const SlotSpec &spec,
    const ClassifierBuffers *classifier, int vocab_tile_size) {
  auto slot = std::make_unique<SlotRuntime>();
  slot->spec = spec;
  slot->engine = &engine;
  if (slot->engine->getNbIOTensors() != 2) {
    throw std::runtime_error("expected exactly two IO tensors in " +
                             spec.engine_path);
  }
  for (int i = 0; i < slot->engine->getNbIOTensors(); ++i) {
    const char *name = slot->engine->getIOTensorName(i);
    const auto mode = slot->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      slot->input_name = name;
    } else {
      slot->output_name = name;
    }
  }
  if (slot->input_name.empty() || slot->output_name.empty()) {
    throw std::runtime_error("failed to discover engine IO names for " +
                             spec.engine_path);
  }

  slot->context.reset(slot->engine->createExecutionContext(
      nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!slot->context) {
    throw std::runtime_error("failed to create execution context for " +
                             spec.name);
  }
  const auto context_memory_bytes = slot->engine->getDeviceMemorySizeV2();
  slot->context_memory_bytes =
      context_memory_bytes > 0 ? static_cast<std::size_t>(context_memory_bytes)
                               : 0;
  slot->context_memory.reset(slot->context_memory_bytes);
  slot->context->setDeviceMemoryV2(slot->context_memory.get(),
                                   context_memory_bytes);

  slot->input_dims = nvinfer1::Dims4{spec.batch, 3, 48, spec.width};
  if (!slot->context->setInputShape(slot->input_name.c_str(),
                                    slot->input_dims)) {
    throw std::runtime_error("failed to set input shape for " + spec.name);
  }
  slot->output_dims = slot->context->getTensorShape(slot->output_name.c_str());
  if (slot->output_dims.nbDims != 3 || slot->output_dims.d[0] != spec.batch ||
      slot->output_dims.d[2] <= 0) {
    throw std::runtime_error("unexpected hidden shape for " + spec.name + ": " +
                             ppocrv6_native::engine::dims_to_string(
                                 slot->output_dims));
  }

  slot->timesteps = slot->output_dims.d[1];
  slot->hidden_size = slot->output_dims.d[2];
  slot->rows = spec.batch * slot->timesteps;
  slot->input_count = static_cast<std::size_t>(spec.batch) * 3U * 48U *
                      static_cast<std::size_t>(spec.width);
  slot->hidden_count = volume(slot->output_dims);
  slot->input_bytes = slot->input_count * sizeof(float);
  slot->hidden_bytes = slot->hidden_count * sizeof(std::uint16_t);

  slot->input.reset(slot->input_count);
  slot->hidden.reset(slot->hidden_count);
  const auto host_input = read_binary<float>(spec.input_path, slot->input_count);
  PPOCRV6_CUDA_CHECK(cudaMemcpy(slot->input.get(), host_input.data(),
                                host_input.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
  slot->context->setTensorAddress(slot->input_name.c_str(), slot->input.get());
  slot->context->setTensorAddress(slot->output_name.c_str(), slot->hidden.get());

  if (classifier != nullptr) {
    if (slot->hidden_size != classifier->hidden_size) {
      throw std::runtime_error("classifier hidden size mismatch for " +
                               spec.name);
    }
    slot->indices.reset(static_cast<std::size_t>(slot->rows));
    slot->prob.reset(static_cast<std::size_t>(slot->rows));
    slot->max_logits.reset(static_cast<std::size_t>(slot->rows));
    slot->workspace = ppocrv6_native::kernels::tiled_classifier_workspace_shape(
        slot->rows, classifier->vocab_size, 16, vocab_tile_size);
    slot->partial_bytes =
        slot->workspace.partial_count *
        (sizeof(int) + sizeof(float) + sizeof(float));
    slot->partial_ids.reset(slot->workspace.partial_count);
    slot->partial_max.reset(slot->workspace.partial_count);
    slot->partial_sum.reset(slot->workspace.partial_count);
  }

  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&slot->stream));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&slot->stop_event));
  return slot;
}

void launch_slot(SlotRuntime &slot, const ClassifierBuffers *classifier,
                 int vocab_tile_size, cudaStream_t stream) {
  if (!slot.context->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT enqueue failed for " + slot.spec.name);
  }
  if (classifier == nullptr) {
    return;
  }
  ppocrv6_native::kernels::cuda_linear_argmax_prob_wmma(
      reinterpret_cast<const half *>(slot.hidden.get()),
      reinterpret_cast<const half *>(classifier->weight.get()),
      reinterpret_cast<const half *>(classifier->bias.get()), slot.indices.get(),
      slot.prob.get(), slot.max_logits.get(), slot.rows, slot.hidden_size,
      classifier->vocab_size, vocab_tile_size, slot.partial_ids.get(),
      slot.partial_max.get(), slot.partial_sum.get(), stream);
}

float time_single_slot(SlotRuntime &slot, const ClassifierBuffers *classifier,
                       int vocab_tile_size, cudaStream_t stream) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  launch_slot(slot, classifier, vocab_tile_size, stream);
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed;
}

float time_sequential_group(
    const std::vector<std::unique_ptr<SlotRuntime>> &slots,
    const ClassifierBuffers *classifier, int vocab_tile_size,
    cudaStream_t stream) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  for (const auto &slot : slots) {
    launch_slot(*slot, classifier, vocab_tile_size, stream);
    PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed;
}

float time_concurrent_group(
    const std::vector<std::unique_ptr<SlotRuntime>> &slots,
    const ClassifierBuffers *classifier, int vocab_tile_size,
    cudaStream_t control_stream) {
  cudaEvent_t start = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, control_stream));
  for (const auto &slot : slots) {
    PPOCRV6_CUDA_CHECK(cudaStreamWaitEvent(slot->stream, start));
    launch_slot(*slot, classifier, vocab_tile_size, slot->stream);
    PPOCRV6_CUDA_CHECK(cudaEventRecord(slot->stop_event, slot->stream));
  }
  float group_ms = 0.0f;
  for (const auto &slot : slots) {
    PPOCRV6_CUDA_CHECK(cudaEventSynchronize(slot->stop_event));
    float slot_ms = 0.0f;
    PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&slot_ms, start, slot->stop_event));
    group_ms = std::max(group_ms, slot_ms);
  }
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  return group_ms;
}

std::vector<float> repeat_times(int repeats, const std::function<float()> &fn) {
  std::vector<float> values;
  values.reserve(static_cast<std::size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    values.push_back(fn());
  }
  return values;
}

void print_distribution(const std::string &prefix,
                        const std::vector<float> &values) {
  std::cout << prefix << "_avg_ms=" << mean(values) << '\n';
  std::cout << prefix << "_min_ms=" << percentile(values, 0.0f) << '\n';
  std::cout << prefix << "_median_ms=" << percentile(values, 0.5f) << '\n';
  std::cout << prefix << "_p95_ms=" << percentile(values, 0.95f) << '\n';
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);
  const std::string mode = arg_or(args, "--mode", "recognition");
  const int warmups = int_arg_or(args, "--warmups", 5);
  const int repeats = int_arg_or(args, "--repeats", 30);
  const int vocab_size = int_arg_or(args, "--vocab-size", 18710);
  const int vocab_tile_size = int_arg_or(args, "--vocab-tile-size", 512);
  const std::size_t max_extra_bytes =
      mib_arg_or(args, "--max-extra-mib", 1400);
  if (mode != "hidden" && mode != "recognition") {
    throw std::runtime_error("--mode must be hidden or recognition");
  }
  if (warmups < 0 || repeats <= 0 || vocab_size <= 0 || vocab_tile_size <= 0) {
    throw std::runtime_error("invalid benchmark arguments");
  }

  std::vector<SlotSpec> specs;
  if (args.slots.empty()) {
    specs.push_back(parse_slot_spec(
        "w640b1,artifacts/ppocrv6-medium/engines/rec-hidden-b1-w640.trt,"
        "tmp/trt-classifier-check/crop0004-w640/input-b1-w640.f32.bin,1,640"));
    specs.push_back(parse_slot_spec(
        "w3200b4,artifacts/ppocrv6-medium/engines/rec-hidden-b4-w3200.trt,"
        "tmp/trt-classifier-check/crops0001-0004-w3200/"
        "input-b4-w3200.f32.bin,4,3200"));
  } else {
    for (const auto &slot : args.slots) {
      specs.push_back(parse_slot_spec(slot));
    }
  }
  if (specs.empty()) {
    throw std::runtime_error("at least one slot is required");
  }

  PPOCRV6_CUDA_CHECK(cudaFree(nullptr));
  const auto memory_after_cuda_context = cuda_memory_snapshot();

  Logger logger;
  auto runtime =
      std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("failed to create TensorRT runtime");
  }
  const auto memory_after_runtime = cuda_memory_snapshot();

  std::map<std::string, std::unique_ptr<nvinfer1::ICudaEngine>> engines;
  for (const auto &spec : specs) {
    if (engines.find(spec.engine_path) != engines.end()) {
      continue;
    }
    const auto engine_bytes = read_file(spec.engine_path);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!engine) {
      throw std::runtime_error("failed to deserialize engine " +
                               spec.engine_path);
    }
    engines.emplace(spec.engine_path, std::move(engine));
  }
  const auto memory_after_engine_load = cuda_memory_snapshot();

  std::unique_ptr<ClassifierBuffers> classifier;
  if (mode == "recognition") {
    classifier = std::make_unique<ClassifierBuffers>();
    classifier->vocab_size = vocab_size;
  }

  std::vector<std::unique_ptr<SlotRuntime>> slots;
  slots.reserve(specs.size());
  for (const auto &spec : specs) {
    slots.push_back(
        create_slot(*engines.at(spec.engine_path), spec, nullptr,
                    vocab_tile_size));
    if (classifier && classifier->hidden_size == 0) {
      classifier->hidden_size = slots.back()->hidden_size;
    }
  }
  const auto memory_after_contexts_and_hidden_buffers = cuda_memory_snapshot();

  if (classifier) {
    const std::string weight_path =
        arg_or(args, "--weight", "artifacts/ppocrv6-medium/classifier/weight.fp16.bin");
    const std::string bias_path =
        arg_or(args, "--bias", "artifacts/ppocrv6-medium/classifier/bias.fp16.bin");
    const std::size_t weight_count =
        static_cast<std::size_t>(classifier->hidden_size) *
        static_cast<std::size_t>(classifier->vocab_size);
    const std::size_t bias_count =
        static_cast<std::size_t>(classifier->vocab_size);
    const auto host_weight = read_binary<std::uint16_t>(weight_path, weight_count);
    const auto host_bias = read_binary<std::uint16_t>(bias_path, bias_count);
    classifier->weight.reset(weight_count);
    classifier->bias.reset(bias_count);
    classifier->weight_bytes = weight_count * sizeof(std::uint16_t);
    classifier->bias_bytes = bias_count * sizeof(std::uint16_t);
    PPOCRV6_CUDA_CHECK(cudaMemcpy(classifier->weight.get(), host_weight.data(),
                                  classifier->weight_bytes,
                                  cudaMemcpyHostToDevice));
    PPOCRV6_CUDA_CHECK(cudaMemcpy(classifier->bias.get(), host_bias.data(),
                                  classifier->bias_bytes,
                                  cudaMemcpyHostToDevice));
    for (auto &slot : slots) {
      slot->compact_output_bytes =
          static_cast<std::size_t>(slot->rows) *
          (sizeof(int) + sizeof(float) + sizeof(float));
      slot->indices.reset(static_cast<std::size_t>(slot->rows));
      slot->prob.reset(static_cast<std::size_t>(slot->rows));
      slot->max_logits.reset(static_cast<std::size_t>(slot->rows));
      slot->workspace = ppocrv6_native::kernels::tiled_classifier_workspace_shape(
          slot->rows, classifier->vocab_size, 16, vocab_tile_size);
      slot->partial_bytes =
          slot->workspace.partial_count *
          (sizeof(int) + sizeof(float) + sizeof(float));
      slot->partial_ids.reset(slot->workspace.partial_count);
      slot->partial_max.reset(slot->workspace.partial_count);
      slot->partial_sum.reset(slot->workspace.partial_count);
    }
  }
  const auto memory_after_all_alloc = cuda_memory_snapshot();
  const std::size_t extra_bytes =
      used_bytes(memory_after_all_alloc) - used_bytes(memory_after_cuda_context);
  if (extra_bytes > max_extra_bytes) {
    std::cerr << "extra_device_memory_mib=" << bytes_to_mib(extra_bytes)
              << " exceeds max_extra_mib=" << bytes_to_mib(max_extra_bytes)
              << '\n';
    return 2;
  }

  cudaStream_t sequential_stream = nullptr;
  cudaStream_t control_stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&sequential_stream));
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&control_stream));

  const ClassifierBuffers *classifier_ptr = classifier.get();
  for (int i = 0; i < warmups; ++i) {
    for (auto &slot : slots) {
      launch_slot(*slot, classifier_ptr, vocab_tile_size, sequential_stream);
    }
  }
  PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(sequential_stream));
  for (int i = 0; i < warmups; ++i) {
    (void)time_concurrent_group(slots, classifier_ptr, vocab_tile_size,
                                control_stream);
  }
  const auto memory_after_warmup = cuda_memory_snapshot();

  std::vector<std::vector<float>> individual_times;
  individual_times.reserve(slots.size());
  for (auto &slot : slots) {
    individual_times.push_back(repeat_times(repeats, [&]() {
      return time_single_slot(*slot, classifier_ptr, vocab_tile_size,
                              sequential_stream);
    }));
  }
  const auto sequential_times = repeat_times(repeats, [&]() {
    return time_sequential_group(slots, classifier_ptr, vocab_tile_size,
                                 sequential_stream);
  });
  const auto concurrent_times = repeat_times(repeats, [&]() {
    return time_concurrent_group(slots, classifier_ptr, vocab_tile_size,
                                 control_stream);
  });

  if (classifier) {
    for (auto &slot : slots) {
      std::vector<int> ids(static_cast<std::size_t>(slot->rows));
      std::vector<float> prob(static_cast<std::size_t>(slot->rows));
      PPOCRV6_CUDA_CHECK(cudaMemcpy(ids.data(), slot->indices.get(),
                                    ids.size() * sizeof(int),
                                    cudaMemcpyDeviceToHost));
      PPOCRV6_CUDA_CHECK(cudaMemcpy(prob.data(), slot->prob.get(),
                                    prob.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost));
      const auto minmax = std::minmax_element(prob.begin(), prob.end());
      slot->prob_min = *minmax.first;
      slot->prob_max = *minmax.second;
      const std::size_t sample_count =
          std::min<std::size_t>(ids.size(), static_cast<std::size_t>(8));
      slot->sample_ids.assign(ids.begin(), ids.begin() + sample_count);
    }
  }

  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(sequential_stream));
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(control_stream));

  std::size_t explicit_slot_bytes = 0;
  for (const auto &slot : slots) {
    explicit_slot_bytes += slot->explicit_bytes();
  }
  const std::size_t classifier_bytes =
      classifier ? classifier->weight_bytes + classifier->bias_bytes : 0;
  const float sequential_median = percentile(sequential_times, 0.5f);
  const float concurrent_median = percentile(concurrent_times, 0.5f);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "trt_multistream_recognition_bench: ok\n";
  std::cout << "mode=" << mode << '\n';
  std::cout << "slot_count=" << slots.size() << '\n';
  std::cout << "engine_count=" << engines.size() << '\n';
  std::cout << "warmups=" << warmups << '\n';
  std::cout << "repeats=" << repeats << '\n';
  std::cout << "vocab_tile_size=" << vocab_tile_size << '\n';
  std::cout << "max_extra_mib=" << bytes_to_mib(max_extra_bytes) << '\n';
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const auto &slot = slots[i];
    std::cout << "slot[" << i << "].name=" << slot->spec.name << '\n';
    std::cout << "slot[" << i << "].engine=" << slot->spec.engine_path << '\n';
    std::cout << "slot[" << i << "].input_shape="
              << ppocrv6_native::engine::dims_to_string(slot->input_dims)
              << '\n';
    std::cout << "slot[" << i << "].hidden_shape="
              << ppocrv6_native::engine::dims_to_string(slot->output_dims)
              << '\n';
    std::cout << "slot[" << i << "].rows=" << slot->rows << '\n';
    std::cout << "slot[" << i << "].context_memory_mib="
              << bytes_to_mib(slot->context_memory_bytes) << '\n';
    std::cout << "slot[" << i << "].input_mib="
              << bytes_to_mib(slot->input_bytes) << '\n';
    std::cout << "slot[" << i << "].hidden_mib="
              << bytes_to_mib(slot->hidden_bytes) << '\n';
    std::cout << "slot[" << i << "].compact_output_mib="
              << bytes_to_mib(slot->compact_output_bytes) << '\n';
    std::cout << "slot[" << i << "].partial_mib="
              << bytes_to_mib(slot->partial_bytes) << '\n';
    std::cout << "slot[" << i << "].explicit_mib="
              << bytes_to_mib(slot->explicit_bytes()) << '\n';
    if (classifier) {
      std::cout << "slot[" << i << "].prob_min=" << slot->prob_min << '\n';
      std::cout << "slot[" << i << "].prob_max=" << slot->prob_max << '\n';
      std::cout << "slot[" << i << "].first_ids=";
      for (std::size_t j = 0; j < slot->sample_ids.size(); ++j) {
        if (j != 0) {
          std::cout << ',';
        }
        std::cout << slot->sample_ids[j];
      }
      std::cout << '\n';
    }
    print_distribution("slot[" + std::to_string(i) + "].individual",
                       individual_times[i]);
  }
  std::cout << "classifier_shared_mib=" << bytes_to_mib(classifier_bytes)
            << '\n';
  std::cout << "explicit_slot_mib=" << bytes_to_mib(explicit_slot_bytes)
            << '\n';
  std::cout << "explicit_total_mib="
            << bytes_to_mib(explicit_slot_bytes + classifier_bytes) << '\n';
  std::cout << "extra_device_memory_since_cuda_context_mib="
            << bytes_to_mib(extra_bytes) << '\n';
  print_memory_snapshot("after_cuda_context", memory_after_cuda_context);
  print_memory_snapshot("after_runtime", memory_after_runtime);
  print_memory_snapshot("after_engine_load", memory_after_engine_load);
  print_memory_snapshot("after_contexts_and_hidden_buffers",
                        memory_after_contexts_and_hidden_buffers);
  print_memory_snapshot("after_all_alloc", memory_after_all_alloc);
  print_memory_snapshot("after_warmup", memory_after_warmup);
  print_distribution("sequential_group", sequential_times);
  print_distribution("concurrent_group", concurrent_times);
  std::cout << "concurrent_speedup_vs_sequential_median="
            << (concurrent_median > 0.0f ? sequential_median / concurrent_median
                                         : 0.0f)
            << '\n';
  return 0;
}
