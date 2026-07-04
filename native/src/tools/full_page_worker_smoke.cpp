#include "ppocrv6_native/full_page/full_page_worker.h"

#include <cstddef>
#include <cstdint>
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

const std::string &required(const std::map<std::string, std::string> &args,
                            const std::string &key) {
  const auto it = args.find(key);
  if (it == args.end()) {
    throw std::runtime_error("missing required argument " + key);
  }
  return it->second;
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

float float_arg_or(const std::map<std::string, std::string> &args,
                   const std::string &key, float fallback) {
  const auto it = args.find(key);
  return it == args.end() ? fallback : std::stof(it->second);
}

std::vector<int> parse_int_csv(const std::string &value) {
  std::vector<int> out;
  std::string current;
  for (char ch : value) {
    if (ch == ',') {
      if (!current.empty()) {
        out.push_back(std::stoi(current));
        current.clear();
      }
    } else {
      current.push_back(ch);
    }
  }
  if (!current.empty()) {
    out.push_back(std::stoi(current));
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

void write_text(const std::string &path, const std::string &value) {
  const auto out_path = std::filesystem::path(path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open output " + path);
  }
  out << value;
  if (!out) {
    throw std::runtime_error("failed to write " + path);
  }
}

} // namespace

int main(int argc, char **argv) {
  const auto args = parse_args(argc, argv);

  ppocrv6_native::full_page::FullPageWorkerConfig config;
  config.detector.engine_path =
      arg_or(args, "--det-engine", config.detector.engine_path);
  config.detector.default_height =
      int_arg_or(args, "--det-default-height", config.detector.default_height);
  config.detector.default_width =
      int_arg_or(args, "--det-default-width", config.detector.default_width);
  config.detector.max_height =
      int_arg_or(args, "--det-max-height", config.detector.max_height);
  config.detector.max_width =
      int_arg_or(args, "--det-max-width", config.detector.max_width);
  config.detector.warmup_runs =
      int_arg_or(args, "--det-warmup-runs", config.detector.warmup_runs);
  config.detector_preprocess.limit_side_len = int_arg_or(
      args, "--det-limit-side-len", config.detector_preprocess.limit_side_len);
  config.detector_preprocess.max_side_limit = int_arg_or(
      args, "--det-max-side-limit", config.detector_preprocess.max_side_limit);
  const auto limit_type = arg_or(args, "--det-limit-type", "max");
  if (limit_type == "max") {
    config.detector_preprocess.limit_type =
        ppocrv6_native::full_page::DetectorLimitType::kMax;
  } else if (limit_type == "min") {
    config.detector_preprocess.limit_type =
        ppocrv6_native::full_page::DetectorLimitType::kMin;
  } else if (limit_type == "resize_long") {
    config.detector_preprocess.limit_type =
        ppocrv6_native::full_page::DetectorLimitType::kResizeLong;
  } else {
    throw std::runtime_error("--det-limit-type must be max, min, or resize_long");
  }

  config.recognizer.engine_path =
      arg_or(args, "--rec-engine", config.recognizer.engine_path);
  config.recognizer.weight_path =
      arg_or(args, "--weight", config.recognizer.weight_path);
  config.recognizer.bias_path = arg_or(args, "--bias", config.recognizer.bias_path);
  config.recognizer.characters_path =
      arg_or(args, "--characters", config.recognizer.characters_path);
  config.recognizer.default_width =
      int_arg_or(args, "--rec-default-width", config.recognizer.default_width);
  config.recognizer.default_batch_size = int_arg_or(
      args, "--rec-default-batch-size", config.recognizer.default_batch_size);
  config.recognizer.max_width =
      int_arg_or(args, "--rec-max-width", config.recognizer.max_width);
  config.recognizer.max_batch_size =
      int_arg_or(args, "--rec-max-batch-size", config.recognizer.max_batch_size);
  config.recognizer.warmup_runs =
      int_arg_or(args, "--rec-warmup-runs", config.recognizer.warmup_runs);

  config.postprocess.thresh =
      float_arg_or(args, "--det-thresh", config.postprocess.thresh);
  config.postprocess.box_thresh =
      float_arg_or(args, "--det-box-thresh", config.postprocess.box_thresh);
  config.postprocess.unclip_ratio = float_arg_or(
      args, "--det-unclip-ratio", config.postprocess.unclip_ratio);
  config.postprocess.max_candidates = int_arg_or(
      args, "--det-max-candidates", config.postprocess.max_candidates);
  config.postprocess.min_size =
      int_arg_or(args, "--det-min-size", config.postprocess.min_size);
  config.recognition_height =
      int_arg_or(args, "--rec-height", config.recognition_height);
  const auto buckets_it = args.find("--rec-buckets");
  if (buckets_it != args.end()) {
    config.recognition_buckets = parse_int_csv(buckets_it->second);
  }

  const int image_height = int_arg_or(args, "--image-height", 0);
  const int image_width = int_arg_or(args, "--image-width", 0);
  const int image_stride =
      int_arg_or(args, "--image-stride", image_width > 0 ? image_width * 3 : 0);
  const int det_height = int_arg_or(args, "--det-height", config.detector.default_height);
  const int det_width = int_arg_or(args, "--det-width", config.detector.default_width);
  if (image_height <= 0 || image_width <= 0 || image_stride < image_width * 3) {
    throw std::runtime_error("invalid image shape arguments");
  }

  const auto image = read_binary<std::uint8_t>(
      required(args, "--image-rgb"),
      static_cast<std::size_t>(image_height) *
          static_cast<std::size_t>(image_stride));
  const auto color_mode = arg_or(args, "--image-color", "rgb");
  if (color_mode != "rgb" && color_mode != "bgr") {
    throw std::runtime_error("--image-color must be rgb or bgr");
  }
  const auto source_color_order =
      color_mode == "bgr" ? ppocrv6_native::full_page::SourceColorOrder::kBgr
                           : ppocrv6_native::full_page::SourceColorOrder::kRgb;

  ppocrv6_native::full_page::FullPageWorker worker(std::move(config));
  std::cout << "info=" << worker.info_json() << '\n';
  ppocrv6_native::full_page::FullPageResult result;
  const auto det_input_it = args.find("--det-input");
  if (det_input_it == args.end()) {
    result = worker.recognize_page_image(image.data(), image_height, image_width,
                                         image_stride, source_color_order);
  } else {
    const auto det_input = read_binary<float>(
        det_input_it->second,
        static_cast<std::size_t>(3) * static_cast<std::size_t>(det_height) *
            static_cast<std::size_t>(det_width));
    result = worker.recognize_page(image.data(), image_height, image_width,
                                   image_stride, source_color_order,
                                   det_input.data(), det_height, det_width);
  }
  const auto json = ppocrv6_native::full_page::full_page_result_to_json(result);
  std::cout << "result=" << json << '\n';

  const auto output_it = args.find("--output-json");
  if (output_it != args.end()) {
    write_text(output_it->second, json);
    std::cout << "output_json=" << output_it->second << '\n';
  }
  return 0;
}
