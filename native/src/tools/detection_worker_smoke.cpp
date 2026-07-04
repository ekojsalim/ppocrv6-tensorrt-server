#include "ppocrv6_native/detection/detection_worker.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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

bool bool_arg_or(const std::map<std::string, std::string> &args,
                 const std::string &key, bool fallback) {
  const auto it = args.find(key);
  if (it == args.end()) {
    return fallback;
  }
  return it->second == "1" || it->second == "true" || it->second == "yes";
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
    throw std::runtime_error("unexpected input file size for " + path);
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

void write_binary_float(const std::string &path,
                        const std::vector<float> &values) {
  const auto out_path = std::filesystem::path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output " + path);
  }
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed to write " + path);
  }
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);
  ppocrv6_native::detection::DetectionWorkerConfig config;
  config.engine_path = arg_or(args, "--engine", config.engine_path);
  config.default_batch = int_arg_or(args, "--default-batch", config.default_batch);
  config.default_height =
      int_arg_or(args, "--default-height", config.default_height);
  config.default_width =
      int_arg_or(args, "--default-width", config.default_width);
  config.max_batch = int_arg_or(args, "--max-batch", config.max_batch);
  config.max_height = int_arg_or(args, "--max-height", config.max_height);
  config.max_width = int_arg_or(args, "--max-width", config.max_width);
  config.warmup_runs = int_arg_or(args, "--warmup-runs", config.warmup_runs);

  const int batch = int_arg_or(args, "--batch", config.default_batch);
  const int height = int_arg_or(args, "--height", config.default_height);
  const int width = int_arg_or(args, "--width", config.default_width);
  const bool copy_output = bool_arg_or(args, "--copy-output", true);
  const std::size_t input_count =
      static_cast<std::size_t>(batch) * 3U * static_cast<std::size_t>(height) *
      static_cast<std::size_t>(width);

  std::vector<float> input;
  const auto input_it = args.find("--input");
  if (input_it == args.end()) {
    input.assign(input_count, 0.0f);
  } else {
    input = read_binary_float(input_it->second, input_count);
  }

  ppocrv6_native::detection::DetectionWorker worker(std::move(config));
  std::cout << "info=" << worker.info_json() << '\n';
  auto result = worker.detect_f32(input.data(), batch, height, width, copy_output);
  std::cout << "result="
            << ppocrv6_native::detection::detection_result_to_json(result)
            << '\n';

  const auto output_it = args.find("--output");
  if (output_it != args.end()) {
    write_binary_float(output_it->second, result.output);
    std::cout << "output=" << output_it->second << '\n';
  }
  return 0;
}
