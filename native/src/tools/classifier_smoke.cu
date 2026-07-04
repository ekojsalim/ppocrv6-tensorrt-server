#include "ppocrv6_native/common/cuda_check.h"
#include "ppocrv6_native/common/cuda_ptr.h"
#include "ppocrv6_native/kernels/classifier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
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

std::string arg_or(const std::map<std::string, std::string> &args,
                   const std::string &key, const std::string &fallback) {
  const auto it = args.find(key);
  return it == args.end() ? fallback : it->second;
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
  const float sum = std::accumulate(values.begin(), values.end(), 0.0f);
  return sum / static_cast<float>(values.size());
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);
  const int batch = int_arg_or(args, "--batch", 1);
  const int timesteps = int_arg_or(args, "--timesteps", 80);
  const int hidden_size = int_arg_or(args, "--hidden-size", 192);
  const int vocab_size = int_arg_or(args, "--vocab-size", 18710);
  const int threads = int_arg_or(args, "--threads", 256);
  const int warmups = int_arg_or(args, "--warmups", 1);
  const int repeats = int_arg_or(args, "--repeats", 1);
  const int row_block_size = int_arg_or(args, "--row-block-size", 8);
  const int vocab_tile_size = int_arg_or(args, "--vocab-tile-size", 256);
  const std::string mode = arg_or(args, "--mode", "simple");
  if (batch <= 0 || timesteps <= 0 || hidden_size <= 0 || vocab_size <= 0 ||
      threads <= 0 || (threads & (threads - 1)) != 0) {
    throw std::runtime_error("invalid classifier shape or thread count");
  }
  if (mode != "simple" && mode != "tiled" && mode != "wmma") {
    throw std::runtime_error("unsupported classifier mode: " + mode);
  }

  const int rows = batch * timesteps;
  const std::size_t hidden_count =
      static_cast<std::size_t>(rows) * hidden_size;
  const std::size_t weight_count =
      static_cast<std::size_t>(hidden_size) * vocab_size;
  const std::size_t bias_count = static_cast<std::size_t>(vocab_size);

  const auto hidden = read_binary<std::uint16_t>(required(args, "--hidden"), hidden_count);
  const auto weight = read_binary<std::uint16_t>(required(args, "--weight"), weight_count);
  const auto bias = read_binary<std::uint16_t>(required(args, "--bias"), bias_count);

  PPOCRV6_CUDA_CHECK(cudaFree(nullptr));
  const auto memory_after_context = cuda_memory_snapshot();

  ppocrv6_native::CudaPtr<std::uint16_t> d_hidden(hidden_count);
  ppocrv6_native::CudaPtr<std::uint16_t> d_weight(weight_count);
  ppocrv6_native::CudaPtr<std::uint16_t> d_bias(bias_count);
  ppocrv6_native::CudaPtr<int> d_indices(static_cast<std::size_t>(rows));
  ppocrv6_native::CudaPtr<float> d_prob(static_cast<std::size_t>(rows));
  ppocrv6_native::CudaPtr<float> d_max_logits(static_cast<std::size_t>(rows));
  ppocrv6_native::CudaPtr<int> d_partial_ids;
  ppocrv6_native::CudaPtr<float> d_partial_max;
  ppocrv6_native::CudaPtr<float> d_partial_sum;
  ppocrv6_native::kernels::TiledClassifierWorkspaceShape tiled_workspace;
  if (mode == "tiled" || mode == "wmma") {
    const int workspace_row_block_size =
        mode == "wmma" ? 16 : row_block_size;
    tiled_workspace = ppocrv6_native::kernels::tiled_classifier_workspace_shape(
        rows, vocab_size, workspace_row_block_size, vocab_tile_size);
    d_partial_ids.reset(tiled_workspace.partial_count);
    d_partial_max.reset(tiled_workspace.partial_count);
    d_partial_sum.reset(tiled_workspace.partial_count);
  }
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

  cudaStream_t stream = nullptr;
  PPOCRV6_CUDA_CHECK(cudaStreamCreate(&stream));
  auto launch_classifier = [&]() {
    if (mode == "tiled") {
      ppocrv6_native::kernels::cuda_linear_argmax_prob_tiled(
          reinterpret_cast<const half *>(d_hidden.get()),
          reinterpret_cast<const half *>(d_weight.get()),
          reinterpret_cast<const half *>(d_bias.get()), d_indices.get(),
          d_prob.get(), d_max_logits.get(), rows, hidden_size, vocab_size,
          row_block_size, vocab_tile_size, threads, d_partial_ids.get(),
          d_partial_max.get(), d_partial_sum.get(), stream);
    } else if (mode == "wmma") {
      ppocrv6_native::kernels::cuda_linear_argmax_prob_wmma(
          reinterpret_cast<const half *>(d_hidden.get()),
          reinterpret_cast<const half *>(d_weight.get()),
          reinterpret_cast<const half *>(d_bias.get()), d_indices.get(),
          d_prob.get(), d_max_logits.get(), rows, hidden_size, vocab_size,
          vocab_tile_size, d_partial_ids.get(), d_partial_max.get(),
          d_partial_sum.get(), stream);
    } else {
      ppocrv6_native::kernels::cuda_linear_argmax_prob(
          reinterpret_cast<const half *>(d_hidden.get()),
          reinterpret_cast<const half *>(d_weight.get()),
          reinterpret_cast<const half *>(d_bias.get()), d_indices.get(),
          d_prob.get(), d_max_logits.get(), rows, hidden_size, vocab_size,
          threads, stream);
    }
  };
  for (int i = 0; i < warmups; ++i) {
    launch_classifier();
  }
  PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto memory_after_warmup = cuda_memory_snapshot();

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&start));
  PPOCRV6_CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> run_ms;
  run_ms.reserve(static_cast<std::size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    PPOCRV6_CUDA_CHECK(cudaEventRecord(start, stream));
    launch_classifier();
    PPOCRV6_CUDA_CHECK(cudaEventRecord(stop, stream));
    PPOCRV6_CUDA_CHECK(cudaEventSynchronize(stop));
    float run_ms_value = 0.0f;
    PPOCRV6_CUDA_CHECK(cudaEventElapsedTime(&run_ms_value, start, stop));
    run_ms.push_back(run_ms_value);
  }
  const float elapsed_ms = std::accumulate(run_ms.begin(), run_ms.end(), 0.0f);
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(start));
  PPOCRV6_CUDA_CHECK(cudaEventDestroy(stop));
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));

  std::vector<int> indices(static_cast<std::size_t>(rows));
  std::vector<float> prob(static_cast<std::size_t>(rows));
  std::vector<float> max_logits(static_cast<std::size_t>(rows));
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
  const auto minmax_logits = std::minmax_element(max_logits.begin(), max_logits.end());
  const std::size_t hidden_bytes = hidden_count * sizeof(std::uint16_t);
  const std::size_t weight_bytes = weight_count * sizeof(std::uint16_t);
  const std::size_t bias_bytes = bias_count * sizeof(std::uint16_t);
  const std::size_t compact_output_bytes =
      static_cast<std::size_t>(rows) *
      (sizeof(int) + sizeof(float) + sizeof(float));
  const std::size_t partial_bytes =
      mode != "simple"
          ? tiled_workspace.partial_count *
                (sizeof(int) + sizeof(float) + sizeof(float))
          : 0;
  const std::size_t device_buffer_bytes =
      hidden_bytes + weight_bytes + bias_bytes + compact_output_bytes +
      partial_bytes;
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "classifier_smoke: rows=" << rows << " hidden_size="
            << hidden_size << " vocab_size=" << vocab_size
            << " threads=" << threads << " mode=" << mode << '\n';
  if (mode == "tiled") {
    std::cout << "row_block_size=" << row_block_size
              << " vocab_tile_size=" << vocab_tile_size
              << " num_row_blocks=" << tiled_workspace.num_row_blocks
              << " num_vocab_blocks=" << tiled_workspace.num_vocab_blocks
              << " partial_count=" << tiled_workspace.partial_count << '\n';
  } else if (mode == "wmma") {
    std::cout << "row_block_size=16"
              << " vocab_tile_size=" << vocab_tile_size
              << " num_row_blocks=" << tiled_workspace.num_row_blocks
              << " num_vocab_blocks=" << tiled_workspace.num_vocab_blocks
              << " partial_count=" << tiled_workspace.partial_count << '\n';
  }
  std::cout << "execute_repeats=" << repeats << '\n';
  std::cout << "execute_total_ms=" << elapsed_ms << '\n';
  std::cout << "execute_avg_ms=" << mean(run_ms) << '\n';
  std::cout << "execute_min_ms=" << percentile(run_ms, 0.0f) << '\n';
  std::cout << "execute_median_ms=" << percentile(run_ms, 0.5f) << '\n';
  std::cout << "execute_p95_ms=" << percentile(run_ms, 0.95f) << '\n';
  std::cout << "prob_min=" << *minmax_prob.first << " prob_max="
            << *minmax_prob.second << '\n';
  std::cout << "max_logit_min=" << *minmax_logits.first << " max_logit_max="
            << *minmax_logits.second << '\n';
  std::cout << "hidden_bytes=" << hidden_bytes << '\n';
  std::cout << "weight_bytes=" << weight_bytes << '\n';
  std::cout << "bias_bytes=" << bias_bytes << '\n';
  std::cout << "compact_output_bytes=" << compact_output_bytes << '\n';
  std::cout << "partial_bytes=" << partial_bytes << '\n';
  std::cout << "device_buffer_bytes=" << device_buffer_bytes << '\n';
  std::cout << "device_buffer_mib=" << bytes_to_mib(device_buffer_bytes)
            << '\n';
  print_memory_snapshot("after_context", memory_after_context);
  print_memory_snapshot("after_alloc", memory_after_alloc);
  print_memory_snapshot("after_warmup", memory_after_warmup);
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
