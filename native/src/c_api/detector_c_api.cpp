#include "ppocrv6_native/c_api/detector.h"

#include "ppocrv6_native/c_api/common.h"
#include "ppocrv6_native/detection/detection_worker.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

struct ppocrv6_detector {
  std::unique_ptr<ppocrv6_native::detection::DetectionWorker> worker;
};

namespace {

using ppocrv6_native::c_api::ffi_guard;
using ppocrv6_native::c_api::json_int_or;
using ppocrv6_native::c_api::json_string_or;
using ppocrv6_native::c_api::set_error;
using ppocrv6_native::c_api::set_output;

ppocrv6_native::detection::DetectionWorkerConfig parse_config(
    const char *config_json) {
  ppocrv6_native::detection::DetectionWorkerConfig config;
  if (config_json == nullptr || config_json[0] == '\0') {
    return config;
  }

  const std::string json(config_json);
  config.engine_path = json_string_or(json, "engine_path", config.engine_path);
  config.default_batch =
      json_int_or(json, "default_batch", config.default_batch);
  config.default_height =
      json_int_or(json, "default_height", config.default_height);
  config.default_width =
      json_int_or(json, "default_width", config.default_width);
  config.max_batch = json_int_or(json, "max_batch", config.max_batch);
  config.max_height = json_int_or(json, "max_height", config.max_height);
  config.max_width = json_int_or(json, "max_width", config.max_width);
  config.warmup_runs = json_int_or(json, "warmup_runs", config.warmup_runs);
  return config;
}

} // namespace

extern "C" int ppocrv6_detector_create(const char *config_json,
                                        ppocrv6_detector **out_handle,
                                        char **out_error) {
  if (out_handle == nullptr) {
    set_error(out_error, "out_handle is null");
    return 1;
  }
  *out_handle = nullptr;
  return ffi_guard(out_error, [&]() {
    auto config = parse_config(config_json);
    auto handle = std::make_unique<ppocrv6_detector>();
    handle->worker =
        std::make_unique<ppocrv6_native::detection::DetectionWorker>(
            std::move(config));
    *out_handle = handle.release();
  });
}

extern "C" void ppocrv6_detector_destroy(ppocrv6_detector *handle) {
  delete handle;
}

extern "C" int ppocrv6_detector_info_json(ppocrv6_detector *handle,
                                           char **out_json, char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "detector handle is null");
    return 1;
  }
  return ffi_guard(out_error,
                   [&]() { set_output(out_json, handle->worker->info_json()); });
}

extern "C" int ppocrv6_detector_detect_f32(
    ppocrv6_detector *handle, const float *nchw, int batch, int height,
    int width, float *out_prob, size_t out_prob_count, char **out_json,
    char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "detector handle is null");
    return 1;
  }
  return ffi_guard(out_error, [&]() {
    const bool copy_output = out_prob != nullptr;
    auto result =
        handle->worker->detect_f32(nchw, batch, height, width, copy_output);
    if (copy_output) {
      if (out_prob_count < result.output.size()) {
        throw std::runtime_error("detector output buffer is too small");
      }
      std::memcpy(out_prob, result.output.data(),
                  result.output.size() * sizeof(float));
    }
    set_output(out_json,
               ppocrv6_native::detection::detection_result_to_json(result));
  });
}

extern "C" void ppocrv6_detector_free_string(char *value) {
  std::free(value);
}
