#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/engine/trt_engine.h"
#include "ppocrv6_native/kernels/classifier.h"
#include "ppocrv6_native/recognition/ctc_decode.h"

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
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Logger final : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT full-rec] " << msg << '\n';
    }
  }
};

struct MemorySnapshot {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

struct ManifestChunk {
  std::string name;
  int bucket_width = 0;
  int capacity = 0;
  int valid_count = 0;
  std::string input_path;
  std::string engine_path;
  std::vector<int> source_indices;
  std::vector<int> natural_widths;
  std::vector<float> host_input;
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
  std::string engine_path;
  nvinfer1::ICudaEngine *engine = nullptr;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  ppocrv6_native::CudaPtr<std::uint8_t> context_memory;
  ppocrv6_native::CudaPtr<float> input;
  ppocrv6_native::CudaPtr<std::uint16_t> hidden;
  ppocrv6_native::CudaPtr<int> indices;
  ppocrv6_native::CudaPtr<float> prob;
  ppocrv6_native::CudaPtr<float> max_logits;
  ppocrv6_native::CudaPtr<int> partial_ids;
  ppocrv6_native::CudaPtr<float> partial_max;
  ppocrv6_native::CudaPtr<float> partial_sum;
  std::string input_name;
  std::string output_name;
  nvinfer1::Dims4 input_dims{};
  nvinfer1::Dims output_dims{};
  ppocrv6_native::kernels::TiledClassifierWorkspaceShape workspace;
  int batch = 0;
  int width = 0;
  int current_profile = -1;
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

  [[nodiscard]] std::size_t explicit_bytes() const {
    return context_memory_bytes + input_bytes + hidden_bytes +
           compact_output_bytes + partial_bytes;
  }
};

struct ChunkTiming {
  float shape_ms = 0.0f;
  float h2d_ms = 0.0f;
  float compute_ms = 0.0f;
  float d2h_ms = 0.0f;
  float ctc_ms = 0.0f;

  [[nodiscard]] float total_ms() const {
    return shape_ms + h2d_ms + compute_ms + d2h_ms + ctc_ms;
  }
};

struct RunTiming {
  float shape_ms = 0.0f;
  float h2d_ms = 0.0f;
  float compute_ms = 0.0f;
  float d2h_ms = 0.0f;
  float ctc_ms = 0.0f;
  float total_ms = 0.0f;
};

struct DecodedRecord {
  int source_index = 0;
  float score = 0.0f;
  std::string text;
};

enum class ContextMode {
  kShape,
  kSwitch,
  kPerProfile,
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

ContextMode parse_context_mode(const std::string &value,
                               bool reuse_single_slot) {
  if (value.empty()) {
    return reuse_single_slot ? ContextMode::kSwitch : ContextMode::kShape;
  }
  if (value == "shape") {
    return ContextMode::kShape;
  }
  if (value == "switch") {
    return ContextMode::kSwitch;
  }
  if (value == "per-profile" || value == "profile") {
    return ContextMode::kPerProfile;
  }
  throw std::runtime_error(
      "--profile-context-mode must be shape, switch, or per-profile");
}

std::string context_mode_name(ContextMode mode) {
  switch (mode) {
  case ContextMode::kShape:
    return "shape";
  case ContextMode::kSwitch:
    return "switch";
  case ContextMode::kPerProfile:
    return "per-profile";
  }
  return "unknown";
}

std::vector<std::string> split(const std::string &value, char delimiter) {
  std::vector<std::string> out;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter)) {
    out.push_back(item);
  }
  return out;
}

std::vector<int> parse_int_list(const std::string &value) {
  std::vector<int> out;
  if (value.empty()) {
    return out;
  }
  for (const auto &part : split(value, ',')) {
    out.push_back(std::stoi(part));
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

std::vector<ManifestChunk> read_manifest(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open manifest " + path);
  }
  std::vector<ManifestChunk> chunks;
  std::string line;
  bool first = true;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    if (first) {
      first = false;
      if (line.rfind("chunk\t", 0) == 0) {
        continue;
      }
    }
    const auto fields = split(line, '\t');
    if (fields.size() != 8) {
      throw std::runtime_error("invalid manifest line: " + line);
    }
    ManifestChunk chunk;
    chunk.name = fields[0];
    chunk.bucket_width = std::stoi(fields[1]);
    chunk.capacity = std::stoi(fields[2]);
    chunk.valid_count = std::stoi(fields[3]);
    chunk.input_path = fields[4];
    chunk.engine_path = fields[5];
    chunk.source_indices = parse_int_list(fields[6]);
    chunk.natural_widths = parse_int_list(fields[7]);
    if (chunk.capacity <= 0 || chunk.bucket_width <= 0 ||
        chunk.valid_count <= 0 || chunk.valid_count > chunk.capacity ||
        static_cast<int>(chunk.source_indices.size()) != chunk.valid_count ||
        static_cast<int>(chunk.natural_widths.size()) != chunk.valid_count) {
      throw std::runtime_error("invalid manifest chunk: " + chunk.name);
    }
    const std::size_t input_count = static_cast<std::size_t>(chunk.capacity) *
                                    3U * 48U *
                                    static_cast<std::size_t>(chunk.bucket_width);
    chunk.host_input = read_binary<float>(chunk.input_path, input_count);
    chunks.push_back(std::move(chunk));
  }
  if (chunks.empty()) {
    throw std::runtime_error("manifest has no chunks: " + path);
  }
  return chunks;
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

std::string clean_tsv_field(std::string value) {
  for (char &ch : value) {
    if (ch == '\t' || ch == '\n' || ch == '\r') {
      ch = ' ';
    }
  }
  return value;
}

std::string slot_key(const ManifestChunk &chunk) {
  return chunk.engine_path + "|b" + std::to_string(chunk.capacity) + "|w" +
         std::to_string(chunk.bucket_width);
}

std::string runtime_slot_key(const ManifestChunk &chunk, ContextMode mode,
                             int profile_index) {
  switch (mode) {
  case ContextMode::kShape:
    return chunk.engine_path + "|p" + std::to_string(profile_index) + "|b" +
           std::to_string(chunk.capacity) + "|w" +
           std::to_string(chunk.bucket_width);
  case ContextMode::kSwitch:
    return chunk.engine_path;
  case ContextMode::kPerProfile:
    return chunk.engine_path + "|p" + std::to_string(profile_index);
  }
  return chunk.engine_path;
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

std::string discover_input_name(nvinfer1::ICudaEngine &engine) {
  for (int i = 0; i < engine.getNbIOTensors(); ++i) {
    const char *name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      return name;
    }
  }
  throw std::runtime_error("failed to discover engine input name");
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

int resolve_profile(nvinfer1::ICudaEngine &engine,
                    const std::string &input_name,
                    const ManifestChunk &chunk) {
  const nvinfer1::Dims4 dims{chunk.capacity, 3, 48, chunk.bucket_width};
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
  throw std::runtime_error("no TensorRT optimization profile accepts " +
                           chunk.name + " input shape " +
                           ppocrv6_native::engine::dims_to_string(dims));
}

std::size_t profile_context_memory_bytes(nvinfer1::ICudaEngine &engine,
                                         int profile_index,
                                         bool allow_switching) {
  const auto bytes =
      allow_switching ? engine.getDeviceMemorySizeV2()
                      : engine.getDeviceMemorySizeForProfileV2(profile_index);
  return bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
}

void set_active_slot_shape(SlotRuntime &slot, const ManifestChunk &chunk,
                           int profile_index,
                           const ClassifierBuffers &classifier,
                           cudaStream_t stream) {
  if (profile_index != slot.current_profile) {
    if (!slot.context->setOptimizationProfileAsync(profile_index, stream)) {
      throw std::runtime_error("failed to set TensorRT optimization profile " +
                               std::to_string(profile_index));
    }
    slot.current_profile = profile_index;
  }
  slot.input_dims = nvinfer1::Dims4{chunk.capacity, 3, 48, chunk.bucket_width};
  if (!slot.context->setInputShape(slot.input_name.c_str(), slot.input_dims)) {
    throw std::runtime_error("failed to set input shape for " + chunk.name);
  }
  slot.output_dims = slot.context->getTensorShape(slot.output_name.c_str());
  if (slot.output_dims.nbDims != 3 ||
      slot.output_dims.d[0] != chunk.capacity || slot.output_dims.d[2] <= 0) {
    throw std::runtime_error("unexpected hidden shape for " + chunk.name + ": " +
                             ppocrv6_native::engine::dims_to_string(
                                 slot.output_dims));
  }

  const auto active_input_count =
      static_cast<std::size_t>(chunk.capacity) * 3U * 48U *
      static_cast<std::size_t>(chunk.bucket_width);
  const auto active_hidden_count = volume(slot.output_dims);
  if (active_input_count * sizeof(float) > slot.input_bytes ||
      active_hidden_count * sizeof(std::uint16_t) > slot.hidden_bytes) {
    throw std::runtime_error("active shape exceeds allocated slot buffers");
  }

  slot.batch = chunk.capacity;
  slot.width = chunk.bucket_width;
  slot.timesteps = slot.output_dims.d[1];
  slot.hidden_size = slot.output_dims.d[2];
  if (slot.hidden_size != classifier.hidden_size) {
    throw std::runtime_error("classifier hidden size mismatch");
  }
  slot.rows = slot.batch * slot.timesteps;
  if (!slot.context->setTensorAddress(slot.input_name.c_str(), slot.input.get()) ||
      !slot.context->setTensorAddress(slot.output_name.c_str(),
                                      slot.hidden.get())) {
    throw std::runtime_error("failed to set TensorRT tensor addresses");
  }
}

std::unique_ptr<SlotRuntime> create_slot(
    nvinfer1::ICudaEngine &engine,
    const std::vector<const ManifestChunk *> &chunks,
    const std::map<std::string, int> &chunk_profiles,
    const ClassifierBuffers &classifier, int vocab_tile_size,
    bool allow_profile_switching) {
  if (chunks.empty()) {
    throw std::runtime_error("cannot create an empty TensorRT slot");
  }
  auto slot = std::make_unique<SlotRuntime>();
  slot->engine_path = chunks.front()->engine_path;
  slot->engine = &engine;

  for (int i = 0; i < slot->engine->getNbIOTensors(); ++i) {
    const char *name = slot->engine->getIOTensorName(i);
    if (slot->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      slot->input_name = name;
    } else {
      slot->output_name = name;
    }
  }
  if (slot->input_name.empty() || slot->output_name.empty()) {
    throw std::runtime_error("failed to discover engine IO names");
  }

  slot->context.reset(slot->engine->createExecutionContext(
      nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!slot->context) {
    throw std::runtime_error("failed to create TensorRT execution context");
  }
  int memory_profile = -1;
  for (const auto *chunk : chunks) {
    const auto profile_it = chunk_profiles.find(chunk->name);
    if (profile_it == chunk_profiles.end()) {
      throw std::runtime_error("missing profile for chunk " + chunk->name);
    }
    if (memory_profile < 0) {
      memory_profile = profile_it->second;
    } else if (memory_profile != profile_it->second) {
      allow_profile_switching = true;
    }
  }
  if (memory_profile < 0) {
    memory_profile = 0;
  }
  slot->context_memory_bytes = profile_context_memory_bytes(
      engine, memory_profile, allow_profile_switching);
  slot->context_memory.reset(slot->context_memory_bytes);
  slot->context->setDeviceMemoryV2(slot->context_memory.get(),
                                   static_cast<int64_t>(slot->context_memory_bytes));

  cudaStream_t probe_stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&probe_stream));
  for (const auto *chunk : chunks) {
    const int profile_index = chunk_profiles.at(chunk->name);
    if (profile_index != slot->current_profile) {
      if (!slot->context->setOptimizationProfileAsync(profile_index,
                                                      probe_stream)) {
        PPOCRV6_CUDA_CHECK(cudaStreamDestroy(probe_stream));
        throw std::runtime_error("failed to set TensorRT profile " +
                                 std::to_string(profile_index) +
                                 " while creating slot");
      }
      slot->current_profile = profile_index;
    }
    slot->input_dims =
        nvinfer1::Dims4{chunk->capacity, 3, 48, chunk->bucket_width};
    if (!slot->context->setInputShape(slot->input_name.c_str(),
                                      slot->input_dims)) {
      PPOCRV6_CUDA_CHECK(cudaStreamDestroy(probe_stream));
      throw std::runtime_error("failed to set input shape for " + chunk->name);
    }
    slot->output_dims =
        slot->context->getTensorShape(slot->output_name.c_str());
    if (slot->output_dims.nbDims != 3 ||
        slot->output_dims.d[0] != chunk->capacity ||
        slot->output_dims.d[2] <= 0) {
      PPOCRV6_CUDA_CHECK(cudaStreamDestroy(probe_stream));
      throw std::runtime_error(
          "unexpected hidden shape for " + chunk->name + ": " +
          ppocrv6_native::engine::dims_to_string(slot->output_dims));
    }
    if (slot->output_dims.d[2] != classifier.hidden_size) {
      PPOCRV6_CUDA_CHECK(cudaStreamDestroy(probe_stream));
      throw std::runtime_error("classifier hidden size mismatch");
    }
    const int timesteps = slot->output_dims.d[1];
    const int rows = chunk->capacity * timesteps;
    const auto input_count =
        static_cast<std::size_t>(chunk->capacity) * 3U * 48U *
        static_cast<std::size_t>(chunk->bucket_width);
    const auto hidden_count = volume(slot->output_dims);
    const auto workspace =
        ppocrv6_native::kernels::tiled_classifier_workspace_shape(
            rows, classifier.vocab_size, 16, vocab_tile_size);

    slot->batch = std::max(slot->batch, chunk->capacity);
    slot->width = std::max(slot->width, chunk->bucket_width);
    slot->timesteps = std::max(slot->timesteps, timesteps);
    slot->hidden_size = classifier.hidden_size;
    slot->rows = std::max(slot->rows, rows);
    slot->input_count = std::max(slot->input_count, input_count);
    slot->hidden_count = std::max(slot->hidden_count, hidden_count);
    slot->workspace.partial_count =
        std::max(slot->workspace.partial_count, workspace.partial_count);
    slot->workspace.num_row_blocks =
        std::max(slot->workspace.num_row_blocks, workspace.num_row_blocks);
    slot->workspace.num_vocab_blocks =
        std::max(slot->workspace.num_vocab_blocks, workspace.num_vocab_blocks);
  }
  PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(probe_stream));
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(probe_stream));

  slot->input_bytes = slot->input_count * sizeof(float);
  slot->hidden_bytes = slot->hidden_count * sizeof(std::uint16_t);
  slot->compact_output_bytes =
      static_cast<std::size_t>(slot->rows) *
      (sizeof(int) + sizeof(float) + sizeof(float));
  slot->partial_bytes =
      slot->workspace.partial_count *
      (sizeof(int) + sizeof(float) + sizeof(float));

  slot->input.reset(slot->input_count);
  slot->hidden.reset(slot->hidden_count);
  slot->indices.reset(static_cast<std::size_t>(slot->rows));
  slot->prob.reset(static_cast<std::size_t>(slot->rows));
  slot->max_logits.reset(static_cast<std::size_t>(slot->rows));
  slot->partial_ids.reset(slot->workspace.partial_count);
  slot->partial_max.reset(slot->workspace.partial_count);
  slot->partial_sum.reset(slot->workspace.partial_count);
  return slot;
}

void launch_compute(SlotRuntime &slot, const ClassifierBuffers &classifier,
                    int vocab_tile_size, cudaStream_t stream) {
  if (!slot.context->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT hidden enqueue failed");
  }
  ppocrv6_native::kernels::cuda_linear_argmax_prob_wmma(
      reinterpret_cast<const half *>(slot.hidden.get()),
      reinterpret_cast<const half *>(classifier.weight.get()),
      reinterpret_cast<const half *>(classifier.bias.get()), slot.indices.get(),
      slot.prob.get(), slot.max_logits.get(), slot.rows, slot.hidden_size,
      classifier.vocab_size, vocab_tile_size, slot.partial_ids.get(),
      slot.partial_max.get(), slot.partial_sum.get(), stream);
}

ChunkTiming run_chunk(
    SlotRuntime &slot, const ManifestChunk &chunk,
    int profile_index,
    const ClassifierBuffers &classifier,
    const std::vector<std::string> &characters, int blank_id,
    int vocab_tile_size, cudaStream_t stream,
    std::vector<DecodedRecord> *last_decoded) {
  ChunkTiming timing;
  const auto shape_start = std::chrono::steady_clock::now();
  set_active_slot_shape(slot, chunk, profile_index, classifier, stream);
  timing.shape_ms = static_cast<float>(
      elapsed_wall_ms(shape_start, std::chrono::steady_clock::now()));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));

  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(slot.input.get(), chunk.host_input.data(),
                                     chunk.host_input.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  timing.h2d_ms = elapsed_event_ms(start, stop);

  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  launch_compute(slot, classifier, vocab_tile_size, stream);
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  timing.compute_ms = elapsed_event_ms(start, stop);

  const std::size_t valid_rows =
      static_cast<std::size_t>(chunk.valid_count) *
      static_cast<std::size_t>(slot.timesteps);
  std::vector<int> ids(valid_rows);
  std::vector<float> prob(valid_rows);
  PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
  PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(ids.data(), slot.indices.get(),
                                     ids.size() * sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
  PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(prob.data(), slot.prob.get(),
                                     prob.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
  PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
  timing.d2h_ms = elapsed_event_ms(start, stop);

  const auto ctc_start = std::chrono::steady_clock::now();
  auto decoded = ppocrv6_native::recognition::ctc_greedy_decode_batch(
      std::span<const int>(ids.data(), ids.size()),
      std::span<const float>(prob.data(), prob.size()), chunk.valid_count,
      slot.timesteps, characters, blank_id);
  const auto ctc_stop = std::chrono::steady_clock::now();
  timing.ctc_ms = std::chrono::duration<float, std::milli>(
                      ctc_stop - ctc_start)
                      .count();
  if (last_decoded != nullptr) {
    for (std::size_t i = 0; i < decoded.size(); ++i) {
      last_decoded->push_back(DecodedRecord{
          chunk.source_indices[i],
          decoded[i].score,
          std::move(decoded[i].text),
      });
    }
  }

  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
  return timing;
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
  const std::string manifest_path =
      arg_or(args, "--manifest", "tmp/native-full-rec/manifest.tsv");
  const std::string weight_path =
      arg_or(args, "--weight", "artifacts/ppocrv6-medium/classifier/weight.fp16.bin");
  const std::string bias_path =
      arg_or(args, "--bias", "artifacts/ppocrv6-medium/classifier/bias.fp16.bin");
  const std::string characters_path =
      arg_or(args, "--characters", "artifacts/ppocrv6-medium/classifier/characters.txt");
  const std::string engine_override = arg_or(args, "--engine-override", "");
  const int vocab_size = int_arg_or(args, "--vocab-size", 18710);
  const int hidden_size = int_arg_or(args, "--hidden-size", 192);
  const int vocab_tile_size = int_arg_or(args, "--vocab-tile-size", 512);
  const int blank_id = int_arg_or(args, "--blank-id", 0);
  const int warmups = int_arg_or(args, "--warmups", 1);
  const int repeats = int_arg_or(args, "--repeats", 5);
  const bool reuse_single_slot = int_arg_or(args, "--reuse-single-slot", 0) != 0;
  const ContextMode context_mode = parse_context_mode(
      arg_or(args, "--profile-context-mode", ""), reuse_single_slot);
  const std::string out_tsv = arg_or(args, "--out-tsv", "");
  const std::size_t max_extra_bytes =
      mib_arg_or(args, "--max-extra-mib", 1400);
  if (vocab_size <= 0 || hidden_size <= 0 || vocab_tile_size <= 0 ||
      warmups < 0 || repeats <= 0) {
    throw std::runtime_error("invalid full recognition benchmark arguments");
  }

  const auto init_total_start = std::chrono::steady_clock::now();
  const auto manifest_start = std::chrono::steady_clock::now();
  auto chunks = read_manifest(manifest_path);
  if (!engine_override.empty()) {
    for (auto &chunk : chunks) {
      chunk.engine_path = engine_override;
    }
  }
  const double init_manifest_ms =
      elapsed_wall_ms(manifest_start, std::chrono::steady_clock::now());
  std::size_t total_valid_crops = 0;
  std::size_t total_padded_rows = 0;
  for (const auto &chunk : chunks) {
    total_valid_crops += static_cast<std::size_t>(chunk.valid_count);
    total_padded_rows += static_cast<std::size_t>(chunk.capacity) *
                         static_cast<std::size_t>(chunk.bucket_width / 8);
  }

  const auto cuda_context_start = std::chrono::steady_clock::now();
  PPOCRV6_CUDA_CHECK(cudaFree(nullptr));
  const double init_cuda_context_ms =
      elapsed_wall_ms(cuda_context_start, std::chrono::steady_clock::now());
  const auto memory_after_cuda_context = cuda_memory_snapshot();

  const auto runtime_start = std::chrono::steady_clock::now();
  Logger logger;
  auto runtime =
      std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    throw std::runtime_error("failed to create TensorRT runtime");
  }
  const double init_runtime_ms =
      elapsed_wall_ms(runtime_start, std::chrono::steady_clock::now());
  const auto memory_after_runtime = cuda_memory_snapshot();

  const auto engine_load_start = std::chrono::steady_clock::now();
  std::map<std::string, std::unique_ptr<nvinfer1::ICudaEngine>> engines;
  for (const auto &chunk : chunks) {
    if (engines.find(chunk.engine_path) != engines.end()) {
      continue;
    }
    const auto engine_bytes = read_file(chunk.engine_path);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
    if (!engine) {
      throw std::runtime_error("failed to deserialize engine " +
                               chunk.engine_path);
    }
    engines.emplace(chunk.engine_path, std::move(engine));
  }
  const double init_engine_load_ms =
      elapsed_wall_ms(engine_load_start, std::chrono::steady_clock::now());
  const auto memory_after_engine_load = cuda_memory_snapshot();

  std::map<std::string, std::string> engine_input_names;
  for (const auto &[path, engine] : engines) {
    engine_input_names.emplace(path, discover_input_name(*engine));
  }
  std::map<std::string, int> chunk_profiles;
  std::map<int, int> profile_chunk_counts;
  for (const auto &chunk : chunks) {
    const int profile = resolve_profile(
        *engines.at(chunk.engine_path),
        engine_input_names.at(chunk.engine_path), chunk);
    chunk_profiles.emplace(chunk.name, profile);
    profile_chunk_counts[profile] += 1;
  }

  const auto classifier_start = std::chrono::steady_clock::now();
  ClassifierBuffers classifier;
  classifier.vocab_size = vocab_size;
  classifier.hidden_size = hidden_size;
  const std::size_t weight_count =
      static_cast<std::size_t>(hidden_size) * static_cast<std::size_t>(vocab_size);
  const std::size_t bias_count = static_cast<std::size_t>(vocab_size);
  const auto host_weight = read_binary<std::uint16_t>(weight_path, weight_count);
  const auto host_bias = read_binary<std::uint16_t>(bias_path, bias_count);
  classifier.weight_bytes = weight_count * sizeof(std::uint16_t);
  classifier.bias_bytes = bias_count * sizeof(std::uint16_t);
  classifier.weight.reset(weight_count);
  classifier.bias.reset(bias_count);
  PPOCRV6_CUDA_CHECK(cudaMemcpy(classifier.weight.get(), host_weight.data(),
                                classifier.weight_bytes,
                                cudaMemcpyHostToDevice));
  PPOCRV6_CUDA_CHECK(cudaMemcpy(classifier.bias.get(), host_bias.data(),
                                classifier.bias_bytes,
                                cudaMemcpyHostToDevice));
  const auto characters = read_characters(characters_path);
  const double init_classifier_ms =
      elapsed_wall_ms(classifier_start, std::chrono::steady_clock::now());

  const auto slot_alloc_start = std::chrono::steady_clock::now();
  std::map<std::string, std::vector<const ManifestChunk *>> slot_groups;
  for (const auto &chunk : chunks) {
    const auto key =
        runtime_slot_key(chunk, context_mode, chunk_profiles.at(chunk.name));
    slot_groups[key].push_back(&chunk);
  }

  std::map<std::string, std::unique_ptr<SlotRuntime>> slots;
  for (const auto &[key, group] : slot_groups) {
    const bool allow_profile_switching =
        context_mode == ContextMode::kSwitch && group.size() > 1;
    slots.emplace(key, create_slot(*engines.at(group.front()->engine_path),
                                   group, chunk_profiles, classifier,
                                   vocab_tile_size, allow_profile_switching));
  }
  const double init_slot_alloc_ms =
      elapsed_wall_ms(slot_alloc_start, std::chrono::steady_clock::now());
  const auto memory_after_all_alloc = cuda_memory_snapshot();
  const std::size_t extra_bytes =
      used_bytes(memory_after_all_alloc) - used_bytes(memory_after_cuda_context);
  if (extra_bytes > max_extra_bytes) {
    std::cerr << "extra_device_memory_mib=" << bytes_to_mib(extra_bytes)
              << " exceeds max_extra_mib=" << bytes_to_mib(max_extra_bytes)
              << '\n';
    return 2;
  }

  const auto warmup_start = std::chrono::steady_clock::now();
  cudaStream_t stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream));
  for (int i = 0; i < warmups; ++i) {
    for (const auto &chunk : chunks) {
      const int profile = chunk_profiles.at(chunk.name);
      auto &slot = *slots.at(runtime_slot_key(chunk, context_mode, profile));
      (void)run_chunk(slot, chunk, profile, classifier, characters, blank_id,
                      vocab_tile_size, stream, nullptr);
    }
  }
  const double init_warmup_ms =
      elapsed_wall_ms(warmup_start, std::chrono::steady_clock::now());
  const auto memory_after_warmup = cuda_memory_snapshot();
  const double init_total_before_repeats_ms =
      elapsed_wall_ms(init_total_start, std::chrono::steady_clock::now());

  std::vector<float> h2d_ms;
  std::vector<float> shape_ms;
  std::vector<float> compute_ms;
  std::vector<float> d2h_ms;
  std::vector<float> ctc_ms;
  std::vector<float> total_ms;
  h2d_ms.reserve(static_cast<std::size_t>(repeats));
  shape_ms.reserve(static_cast<std::size_t>(repeats));
  compute_ms.reserve(static_cast<std::size_t>(repeats));
  d2h_ms.reserve(static_cast<std::size_t>(repeats));
  ctc_ms.reserve(static_cast<std::size_t>(repeats));
  total_ms.reserve(static_cast<std::size_t>(repeats));
  std::vector<DecodedRecord> last_decoded;

  for (int repeat = 0; repeat < repeats; ++repeat) {
    RunTiming run;
    std::vector<DecodedRecord> decoded;
    decoded.reserve(total_valid_crops);
    for (const auto &chunk : chunks) {
      const int profile = chunk_profiles.at(chunk.name);
      auto &slot = *slots.at(runtime_slot_key(chunk, context_mode, profile));
      const auto timing = run_chunk(
          slot, chunk, profile, classifier, characters, blank_id, vocab_tile_size,
          stream, repeat == repeats - 1 ? &decoded : nullptr);
      run.shape_ms += timing.shape_ms;
      run.h2d_ms += timing.h2d_ms;
      run.compute_ms += timing.compute_ms;
      run.d2h_ms += timing.d2h_ms;
      run.ctc_ms += timing.ctc_ms;
    }
    run.total_ms =
        run.shape_ms + run.h2d_ms + run.compute_ms + run.d2h_ms + run.ctc_ms;
    shape_ms.push_back(run.shape_ms);
    h2d_ms.push_back(run.h2d_ms);
    compute_ms.push_back(run.compute_ms);
    d2h_ms.push_back(run.d2h_ms);
    ctc_ms.push_back(run.ctc_ms);
    total_ms.push_back(run.total_ms);
    if (repeat == repeats - 1) {
      last_decoded = std::move(decoded);
    }
  }
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));

  std::sort(last_decoded.begin(), last_decoded.end(),
            [](const DecodedRecord &a, const DecodedRecord &b) {
              return a.source_index < b.source_index;
            });
  if (!out_tsv.empty()) {
    std::ofstream out(out_tsv);
    if (!out) {
      throw std::runtime_error("failed to open output TSV " + out_tsv);
    }
    out << "source_index\tscore\ttext\n";
    for (const auto &record : last_decoded) {
      out << record.source_index << '\t' << record.score << '\t'
          << clean_tsv_field(record.text) << '\n';
    }
  }

  std::size_t explicit_slot_bytes = 0;
  for (const auto &entry : slots) {
    explicit_slot_bytes += entry.second->explicit_bytes();
  }
  const std::size_t classifier_bytes =
      classifier.weight_bytes + classifier.bias_bytes;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "trt_full_recognition_bench: ok\n";
  std::cout << "manifest=" << manifest_path << '\n';
  if (!engine_override.empty()) {
    std::cout << "engine_override=" << engine_override << '\n';
  }
  std::cout << "chunk_count=" << chunks.size() << '\n';
  std::cout << "engine_count=" << engines.size() << '\n';
  std::cout << "slot_count=" << slots.size() << '\n';
  std::cout << "valid_crops=" << total_valid_crops << '\n';
  std::cout << "padded_rows=" << total_padded_rows << '\n';
  std::cout << "warmups=" << warmups << '\n';
  std::cout << "repeats=" << repeats << '\n';
  std::cout << "reuse_single_slot=" << (reuse_single_slot ? 1 : 0) << '\n';
  std::cout << "profile_context_mode=" << context_mode_name(context_mode)
            << '\n';
  for (const auto &[profile, count] : profile_chunk_counts) {
    std::cout << "profile[" << profile << "].chunk_count=" << count << '\n';
  }
  std::cout << "vocab_tile_size=" << vocab_tile_size << '\n';
  std::cout << "init_manifest_ms=" << init_manifest_ms << '\n';
  std::cout << "init_cuda_context_ms=" << init_cuda_context_ms << '\n';
  std::cout << "init_runtime_ms=" << init_runtime_ms << '\n';
  std::cout << "init_engine_load_ms=" << init_engine_load_ms << '\n';
  std::cout << "init_classifier_ms=" << init_classifier_ms << '\n';
  std::cout << "init_slot_alloc_ms=" << init_slot_alloc_ms << '\n';
  std::cout << "init_warmup_ms=" << init_warmup_ms << '\n';
  std::cout << "init_total_before_repeats_ms="
            << init_total_before_repeats_ms << '\n';
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
  print_memory_snapshot("after_all_alloc", memory_after_all_alloc);
  print_memory_snapshot("after_warmup", memory_after_warmup);
  print_distribution("shape", shape_ms);
  print_distribution("h2d", h2d_ms);
  print_distribution("compute", compute_ms);
  print_distribution("d2h", d2h_ms);
  print_distribution("ctc", ctc_ms);
  print_distribution("total", total_ms);
  std::cout << "decoded_count=" << last_decoded.size() << '\n';
  for (std::size_t i = 0; i < std::min<std::size_t>(last_decoded.size(), 8);
       ++i) {
    std::cout << "decoded[" << i << "].source_index="
              << last_decoded[i].source_index << '\n';
    std::cout << "decoded[" << i << "].score=" << last_decoded[i].score
              << '\n';
    std::cout << "decoded[" << i << "].text=" << last_decoded[i].text
              << '\n';
  }
  return 0;
}
