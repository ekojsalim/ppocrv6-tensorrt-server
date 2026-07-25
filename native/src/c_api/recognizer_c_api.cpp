#include "ppocrv6_native/c_api/recognizer.h"

#include "ppocrv6_native/c_api/common.h"
#include "ppocrv6_native/recognition/recognition_worker.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

struct ppocrv6_recognizer {
  std::unique_ptr<ppocrv6_native::recognition::RecognitionWorker> worker;
};

namespace {

using ppocrv6_native::c_api::ffi_guard;
using ppocrv6_native::c_api::json_int_or;
using ppocrv6_native::c_api::json_string_or;
using ppocrv6_native::c_api::set_error;
using ppocrv6_native::c_api::set_output;

ppocrv6_native::recognition::RecognitionWorkerConfig parse_config(
    const char *config_json) {
  ppocrv6_native::recognition::RecognitionWorkerConfig config;
  if (config_json == nullptr || config_json[0] == '\0') {
    return config;
  }

  const std::string json(config_json);
  config.engine_path = json_string_or(json, "engine_path", config.engine_path);
  config.weight_path = json_string_or(json, "weight_path", config.weight_path);
  config.bias_path = json_string_or(json, "bias_path", config.bias_path);
  config.characters_path =
      json_string_or(json, "characters_path", config.characters_path);
  config.default_width =
      json_int_or(json, "default_width", config.default_width);
  config.default_batch_size =
      json_int_or(json, "default_batch_size", config.default_batch_size);
  config.max_batch_size =
      json_int_or(json, "max_batch_size", config.max_batch_size);
  config.max_width = json_int_or(json, "max_width", config.max_width);
  config.vocab_size = json_int_or(json, "vocab_size", config.vocab_size);
  config.hidden_size = json_int_or(json, "hidden_size", config.hidden_size);
  config.vocab_tile_size =
      json_int_or(json, "vocab_tile_size", config.vocab_tile_size);
  config.blank_id = json_int_or(json, "blank_id", config.blank_id);
  config.warmup_runs = json_int_or(json, "warmup_runs", config.warmup_runs);
  return config;
}

ppocrv6_native::recognition::CharacterPolicy parse_character_policy(
    int character_policy) {
  switch (character_policy) {
  case PPOCRV6_CHARACTER_POLICY_ALL:
    return ppocrv6_native::recognition::CharacterPolicy::kAll;
  case PPOCRV6_CHARACTER_POLICY_SUPPRESS_ASCII:
    return ppocrv6_native::recognition::CharacterPolicy::kSuppressAscii;
  case PPOCRV6_CHARACTER_POLICY_CJK_FOCUS:
    return ppocrv6_native::recognition::CharacterPolicy::kCjkFocus;
  case PPOCRV6_CHARACTER_POLICY_CJK_FOCUS_FALLBACK:
    return ppocrv6_native::recognition::CharacterPolicy::kCjkFocusFallback;
  default:
    throw std::invalid_argument("unsupported character policy");
  }
}

} // namespace

extern "C" int ppocrv6_recognizer_create(const char *config_json,
                                          ppocrv6_recognizer **out_handle,
                                          char **out_error) {
  if (out_handle == nullptr) {
    set_error(out_error, "out_handle is null");
    return 1;
  }
  *out_handle = nullptr;
  return ffi_guard(out_error, [&]() {
    auto config = parse_config(config_json);
    auto handle = std::make_unique<ppocrv6_recognizer>();
    handle->worker =
        std::make_unique<ppocrv6_native::recognition::RecognitionWorker>(
            std::move(config));
    *out_handle = handle.release();
  });
}

extern "C" void ppocrv6_recognizer_destroy(ppocrv6_recognizer *handle) {
  delete handle;
}

extern "C" int ppocrv6_recognizer_info_json(ppocrv6_recognizer *handle,
                                             char **out_json,
                                             char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "recognizer handle is null");
    return 1;
  }
  return ffi_guard(out_error,
                   [&]() { set_output(out_json, handle->worker->info_json()); });
}

extern "C" int ppocrv6_recognizer_recognize_f32(
    ppocrv6_recognizer *handle, const float *nchw, int count, int width,
    int batch_size, int return_timesteps, char **out_json, char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "recognizer handle is null");
    return 1;
  }
  return ffi_guard(out_error, [&]() {
    const bool include_timesteps = return_timesteps != 0;
    auto result = handle->worker->recognize_f32(
        nchw, count, width, batch_size, include_timesteps);
    set_output(out_json, ppocrv6_native::recognition::recognition_result_to_json(
                              result, include_timesteps));
  });
}

extern "C" int ppocrv6_recognizer_recognize_f32_with_options(
    ppocrv6_recognizer *handle, const float *nchw, int count, int width,
    int batch_size, int return_timesteps, int character_policy,
    char **out_json, char **out_error) {
  if (handle == nullptr || handle->worker == nullptr) {
    set_error(out_error, "recognizer handle is null");
    return 1;
  }
  return ffi_guard(out_error, [&]() {
    const bool include_timesteps = return_timesteps != 0;
    auto result = handle->worker->recognize_f32(
        nchw, count, width, batch_size, include_timesteps,
        parse_character_policy(character_policy));
    set_output(out_json, ppocrv6_native::recognition::recognition_result_to_json(
                              result, include_timesteps));
  });
}

extern "C" void ppocrv6_recognizer_free_string(char *value) {
  std::free(value);
}
