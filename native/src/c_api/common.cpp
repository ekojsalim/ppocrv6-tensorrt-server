#include "ppocrv6_native/c_api/common.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace ppocrv6_native::c_api {
namespace {

std::size_t find_key(const std::string &json, const std::string &key) {
  const std::string quoted = "\"" + key + "\"";
  const auto pos = json.find(quoted);
  if (pos == std::string::npos) {
    return std::string::npos;
  }
  auto colon = json.find(':', pos + quoted.size());
  if (colon == std::string::npos) {
    return std::string::npos;
  }
  ++colon;
  while (colon < json.size() &&
         std::isspace(static_cast<unsigned char>(json[colon])) != 0) {
    ++colon;
  }
  return colon;
}

std::string parse_json_string_value(const std::string &json, std::size_t pos) {
  if (pos >= json.size() || json[pos] != '"') {
    throw std::runtime_error("expected JSON string value");
  }
  ++pos;
  std::string out;
  while (pos < json.size()) {
    const char ch = json[pos++];
    if (ch == '"') {
      return out;
    }
    if (ch != '\\') {
      out.push_back(ch);
      continue;
    }
    if (pos >= json.size()) {
      throw std::runtime_error("unterminated JSON escape");
    }
    const char escaped = json[pos++];
    switch (escaped) {
    case '"':
    case '\\':
    case '/':
      out.push_back(escaped);
      break;
    case 'b':
      out.push_back('\b');
      break;
    case 'f':
      out.push_back('\f');
      break;
    case 'n':
      out.push_back('\n');
      break;
    case 'r':
      out.push_back('\r');
      break;
    case 't':
      out.push_back('\t');
      break;
    default:
      throw std::runtime_error("unsupported JSON escape in config");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

} // namespace

char *copy_string(const std::string &value) {
  auto *out = static_cast<char *>(std::malloc(value.size() + 1));
  if (out == nullptr) {
    return nullptr;
  }
  std::memcpy(out, value.data(), value.size());
  out[value.size()] = '\0';
  return out;
}

void set_error(char **out_error, const std::string &message) {
  if (out_error != nullptr) {
    *out_error = copy_string(message);
  }
}

void set_output(char **out, const std::string &value) {
  if (out == nullptr) {
    throw std::runtime_error("output string pointer is null");
  }
  *out = copy_string(value);
  if (*out == nullptr) {
    throw std::runtime_error("failed to allocate output string");
  }
}

std::string json_string_or(const std::string &json, const std::string &key,
                           const std::string &fallback) {
  const auto pos = find_key(json, key);
  if (pos == std::string::npos) {
    return fallback;
  }
  return parse_json_string_value(json, pos);
}

int json_int_or(const std::string &json, const std::string &key,
                int fallback) {
  const auto pos = find_key(json, key);
  if (pos == std::string::npos) {
    return fallback;
  }
  std::size_t consumed = 0;
  const int value = std::stoi(json.substr(pos), &consumed);
  if (consumed == 0) {
    throw std::runtime_error("expected JSON integer value for " + key);
  }
  return value;
}

float json_float_or(const std::string &json, const std::string &key,
                    float fallback) {
  const auto pos = find_key(json, key);
  if (pos == std::string::npos) {
    return fallback;
  }
  std::size_t consumed = 0;
  const float value = std::stof(json.substr(pos), &consumed);
  if (consumed == 0) {
    throw std::runtime_error("expected JSON float value for " + key);
  }
  return value;
}

std::vector<int> parse_int_csv(const std::string &value) {
  std::vector<int> out;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      continue;
    }
    out.push_back(std::stoi(item));
  }
  return out;
}

} // namespace ppocrv6_native::c_api
