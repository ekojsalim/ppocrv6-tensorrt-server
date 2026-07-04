#pragma once

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace ppocrv6_native::c_api {

char *copy_string(const std::string &value);
void set_error(char **out_error, const std::string &message);
void set_output(char **out, const std::string &value);

std::string json_string_or(const std::string &json, const std::string &key,
                           const std::string &fallback);
int json_int_or(const std::string &json, const std::string &key, int fallback);
float json_float_or(const std::string &json, const std::string &key,
                    float fallback);
std::vector<int> parse_int_csv(const std::string &value);

template <typename Func>
int ffi_guard(char **out_error, Func &&func,
              const char *unknown_message = "unknown native C API error") {
  try {
    func();
    return 0;
  } catch (const std::exception &exc) {
    set_error(out_error, exc.what());
    return 1;
  } catch (...) {
    set_error(out_error, unknown_message);
    return 1;
  }
}

} // namespace ppocrv6_native::c_api
