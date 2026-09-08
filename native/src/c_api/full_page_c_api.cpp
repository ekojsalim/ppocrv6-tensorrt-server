#include "ppocrv6_native/c_api/full_page.h"

#include "ppocrv6_native/c_api/common.h"
#include "ppocrv6_native/full_page/full_page_worker.h"

#include "recognizer_handle.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

struct ppocrv6_full_page {
  std::unique_ptr<ppocrv6_native::full_page::FullPageWorker> worker;
};

namespace {

using ppocrv6_native::c_api::ffi_guard;
using ppocrv6_native::c_api::json_float_or;
using ppocrv6_native::c_api::json_int_or;
using ppocrv6_native::c_api::json_string_or;
using ppocrv6_native::c_api::parse_int_csv;
using ppocrv6_native::c_api::set_error;
using ppocrv6_native::c_api::set_output;

ppocrv6_native::full_page::DetectorLimitType
parse_limit_type(const std::string &value) {
  if (value == "max") {
    return ppocrv6_native::full_page::DetectorLimitType::kMax;
  }
  if (value == "min") {
    return ppocrv6_native::full_page::DetectorLimitType::kMin;
  }
  if (value == "resize_long") {
    return ppocrv6_native::full_page::DetectorLimitType::kResizeLong;
  }
  throw std::runtime_error(
      "detector_limit_type must be one of max, min, resize_long");
}

ppocrv6_native::full_page::SourceColorOrder
parse_source_color_order(int source_color_order) {
  return source_color_order == 1
             ? ppocrv6_native::full_page::SourceColorOrder::kBgr
             : ppocrv6_native::full_page::SourceColorOrder::kRgb;
}

ppocrv6_native::full_page::FullPageWorkerConfig
parse_config(const char *config_json) {
  ppocrv6_native::full_page::FullPageWorkerConfig config;
  if (config_json == nullptr || config_json[0] == '\0') {
    return config;
  }

  const std::string json(config_json);
  config.detector.engine_path =
      json_string_or(json, "detector_engine_path", config.detector.engine_path);
  config.detector.default_batch =
      json_int_or(json, "detector_default_batch", config.detector.default_batch);
  config.detector.default_height = json_int_or(
      json, "detector_default_height", config.detector.default_height);
  config.detector.default_width =
      json_int_or(json, "detector_default_width", config.detector.default_width);
  config.detector.max_batch =
      json_int_or(json, "detector_max_batch", config.detector.max_batch);
  config.detector.max_height =
      json_int_or(json, "detector_max_height", config.detector.max_height);
  config.detector.max_width =
      json_int_or(json, "detector_max_width", config.detector.max_width);
  config.detector.warmup_runs =
      json_int_or(json, "detector_warmup_runs", config.detector.warmup_runs);
  config.detector_preprocess.limit_side_len =
      json_int_or(json, "detector_limit_side_len",
                  config.detector_preprocess.limit_side_len);
  config.detector_preprocess.max_side_limit =
      json_int_or(json, "detector_max_side_limit",
                  config.detector_preprocess.max_side_limit);
  config.detector_preprocess.limit_type = parse_limit_type(json_string_or(
      json, "detector_limit_type", "max"));

  config.recognizer.engine_path = json_string_or(
      json, "recognizer_engine_path", config.recognizer.engine_path);
  config.recognizer.weight_path =
      json_string_or(json, "weight_path", config.recognizer.weight_path);
  config.recognizer.bias_path =
      json_string_or(json, "bias_path", config.recognizer.bias_path);
  config.recognizer.characters_path = json_string_or(
      json, "characters_path", config.recognizer.characters_path);
  config.recognizer.default_width = json_int_or(
      json, "recognizer_default_width", config.recognizer.default_width);
  config.recognizer.default_batch_size =
      json_int_or(json, "recognizer_default_batch_size",
                  config.recognizer.default_batch_size);
  config.recognizer.max_batch_size = json_int_or(
      json, "recognizer_max_batch_size", config.recognizer.max_batch_size);
  config.recognizer.max_width =
      json_int_or(json, "recognizer_max_width", config.recognizer.max_width);
  config.recognizer.vocab_size =
      json_int_or(json, "vocab_size", config.recognizer.vocab_size);
  config.recognizer.hidden_size =
      json_int_or(json, "hidden_size", config.recognizer.hidden_size);
  config.recognizer.vocab_tile_size =
      json_int_or(json, "vocab_tile_size", config.recognizer.vocab_tile_size);
  config.recognizer.blank_id =
      json_int_or(json, "blank_id", config.recognizer.blank_id);
  config.recognizer.warmup_runs =
      json_int_or(json, "recognizer_warmup_runs", config.recognizer.warmup_runs);

  config.postprocess.thresh =
      json_float_or(json, "det_thresh", config.postprocess.thresh);
  config.postprocess.box_thresh =
      json_float_or(json, "det_box_thresh", config.postprocess.box_thresh);
  config.postprocess.unclip_ratio = json_float_or(
      json, "det_unclip_ratio", config.postprocess.unclip_ratio);
  config.postprocess.max_candidates = json_int_or(
      json, "det_max_candidates", config.postprocess.max_candidates);
  config.postprocess.min_size =
      json_int_or(json, "det_min_size", config.postprocess.min_size);

  config.recognition_height =
      json_int_or(json, "recognition_height", config.recognition_height);
  const auto buckets = json_string_or(json, "recognition_buckets_csv", "");
  if (!buckets.empty()) {
    config.recognition_buckets = parse_int_csv(buckets);
  }
  return config;
}

} // namespace

extern "C" int ppocrv6_full_page_create(const char *config_json,
                                         ppocrv6_full_page **out_handle,
                                         char **out_error) {
  if (out_handle == nullptr) {
    set_error(out_error, "out_handle is null");
    return 1;
  }
  *out_handle = nullptr;
  return ffi_guard(out_error, [&]() {
    auto config = parse_config(config_json);
    auto handle = std::make_unique<ppocrv6_full_page>();
    handle->worker =
        std::make_unique<ppocrv6_native::full_page::FullPageWorker>(
            std::move(config));
    *out_handle = handle.release();
  });
}

extern "C" void ppocrv6_full_page_destroy(ppocrv6_full_page *handle) {
  delete handle;
}

extern "C" int ppocrv6_full_page_info_json(ppocrv6_full_page *handle,
                                            char **out_json,
                                            char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "full-page handle is null");
    return 1;
  }
  return ffi_guard(out_error,
                   [&]() { set_output(out_json, handle->worker->info_json()); });
}

extern "C" int ppocrv6_full_page_recognize(
    ppocrv6_full_page *handle, const unsigned char *image, int image_height,
    int image_width, int image_stride, int source_color_order,
    const float *detector_nchw, int detector_height, int detector_width,
    char **out_json, char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "full-page handle is null");
    return 1;
  }
  return ffi_guard(out_error, [&]() {
    const auto color_order = parse_source_color_order(source_color_order);
    auto result = handle->worker->recognize_page(
        image, image_height, image_width, image_stride, color_order,
        detector_nchw, detector_height, detector_width);
    set_output(out_json,
               ppocrv6_native::full_page::full_page_result_to_json(result));
  });
}

extern "C" int ppocrv6_full_page_recognize_image(
    ppocrv6_full_page *handle, const unsigned char *image, int image_height,
    int image_width, int image_stride, int source_color_order, char **out_json,
    char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "full-page handle is null");
    return 1;
  }
  return ffi_guard(out_error, [&]() {
    const auto color_order = parse_source_color_order(source_color_order);
    auto result = handle->worker->recognize_page_image(
        image, image_height, image_width, image_stride, color_order);
    set_output(out_json,
               ppocrv6_native::full_page::full_page_result_to_json(result));
  });
}

extern "C" void ppocrv6_full_page_free_string(char *value) {
  std::free(value);
}

extern "C" int ppocrv6_full_page_create_with_recognizer(const char *config_json,
    ppocrv6_recognizer *recognizer, ppocrv6_full_page **out_handle, char **out_error) {
  if (!out_handle) { set_error(out_error, "out_handle is null"); return 1; }
  *out_handle = nullptr;
  if (!recognizer || !recognizer->worker) {
    set_error(out_error, "recognizer handle is null"); return 1;
  }
  return ffi_guard(out_error, [&]() {
    auto handle = std::make_unique<ppocrv6_full_page>();
    handle->worker = std::make_unique<ppocrv6_native::full_page::FullPageWorker>(
        parse_config(config_json), recognizer->worker);
    *out_handle = handle.release();
  });
}
