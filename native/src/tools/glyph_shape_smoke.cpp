#include "ppocrv6_native/recognition/glyph_shape.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kHeight = 48;
constexpr int kWidth = 128;

std::vector<float> white_tensor() {
  return std::vector<float>(3U * kHeight * kWidth, 1.0f);
}

void set_ink(std::vector<float> &tensor, int x, int y, float value = -1.0f) {
  const std::size_t plane = static_cast<std::size_t>(kHeight) * kWidth;
  const std::size_t offset = static_cast<std::size_t>(y) * kWidth + x;
  for (int channel = 0; channel < 3; ++channel) {
    tensor[static_cast<std::size_t>(channel) * plane + offset] = value;
  }
}

void expect(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  auto horizontal = white_tensor();
  for (int y = 23; y <= 25; ++y) {
    for (int x = 18; x <= 109; ++x) {
      set_ink(horizontal, x, y);
    }
  }
  const auto horizontal_features =
      ppocrv6_native::recognition::detect_long_horizontal_stroke(
          horizontal, kHeight, kWidth);
  expect(horizontal_features.matched, "long horizontal stroke was rejected");

  auto short_horizontal = white_tensor();
  for (int x = 48; x <= 79; ++x) {
    set_ink(short_horizontal, x, 24);
  }
  expect(ppocrv6_native::recognition::detect_long_horizontal_stroke(
             short_horizontal, kHeight, kWidth)
             .matched,
         "short horizontal stroke was rejected");

  auto faint_horizontal = white_tensor();
  for (int x = 24; x <= 103; ++x) {
    set_ink(faint_horizontal, x, 24, 0.35f);
  }
  expect(ppocrv6_native::recognition::detect_long_horizontal_stroke(
             faint_horizontal, kHeight, kWidth)
             .matched,
         "faint horizontal stroke was rejected");

  auto diagonal = white_tensor();
  for (int x = 28; x <= 99; ++x) {
    const int y = 36 - (x - 28) / 3;
    for (int delta = -1; delta <= 1; ++delta) {
      set_ink(diagonal, x, std::clamp(y + delta, 0, kHeight - 1));
    }
  }
  expect(!ppocrv6_native::recognition::detect_long_horizontal_stroke(
              diagonal, kHeight, kWidth)
              .matched,
         "diagonal stroke was accepted");

  auto vertical = white_tensor();
  for (int y = 7; y <= 40; ++y) {
    for (int x = 62; x <= 65; ++x) {
      set_ink(vertical, x, y);
    }
  }
  expect(!ppocrv6_native::recognition::detect_long_horizontal_stroke(
              vertical, kHeight, kWidth)
              .matched,
         "vertical stroke was accepted");

  const auto blank = white_tensor();
  expect(!ppocrv6_native::recognition::detect_long_horizontal_stroke(
              blank, kHeight, kWidth)
              .matched,
         "blank image was accepted");

  auto two_lines = white_tensor();
  for (int y : {15, 16, 31, 32}) {
    for (int x = 20; x <= 107; ++x) {
      set_ink(two_lines, x, y);
    }
  }
  expect(!ppocrv6_native::recognition::detect_long_horizontal_stroke(
              two_lines, kHeight, kWidth)
              .matched,
         "two-line glyph was accepted as one stroke");

  std::cout << "PASS glyph shape smoke width_ratio="
            << horizontal_features.width_ratio
            << " aspect_ratio=" << horizontal_features.aspect_ratio << '\n';
  return 0;
}
