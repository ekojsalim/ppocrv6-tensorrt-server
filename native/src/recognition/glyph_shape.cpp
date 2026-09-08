#include "ppocrv6_native/recognition/glyph_shape.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ppocrv6_native::recognition {
namespace {

constexpr float kActiveBackgroundMin = 0.75f;
constexpr float kInkDarknessMin = 0.20f;
constexpr float kMinWidthRatio = 0.20f;
constexpr float kMinAspectRatio = 8.0f;
constexpr float kMaxHeightRatio = 0.20f;
constexpr float kMinInkColumnCoverage = 0.80f;
constexpr int kMinBboxWidth = 12;

float grayscale_at(std::span<const float> nchw, int height, int width, int y,
                   int x) {
  const std::size_t plane =
      static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  const std::size_t offset =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
      static_cast<std::size_t>(x);
  return (nchw[offset] + nchw[plane + offset] + nchw[2U * plane + offset]) /
         3.0f;
}

} // namespace

HorizontalStrokeFeatures detect_long_horizontal_stroke(
    std::span<const float> nchw, int height, int width) {
  if (height <= 0 || width <= 0) {
    throw std::invalid_argument("glyph shape dimensions must be positive");
  }
  const std::size_t expected =
      3U * static_cast<std::size_t>(height) *
      static_cast<std::size_t>(width);
  if (nchw.size() != expected) {
    throw std::invalid_argument("glyph shape tensor does not match NCHW shape");
  }

  HorizontalStrokeFeatures features;
  for (int x = width - 1; x >= 0; --x) {
    bool contains_background = false;
    for (int y = 0; y < height; ++y) {
      if (grayscale_at(nchw, height, width, y, x) >=
          kActiveBackgroundMin) {
        contains_background = true;
        break;
      }
    }
    if (contains_background) {
      features.active_width = x + 1;
      break;
    }
  }
  if (features.active_width == 0) {
    return features;
  }

  int min_x = std::numeric_limits<int>::max();
  int max_x = -1;
  int min_y = std::numeric_limits<int>::max();
  int max_y = -1;
  std::vector<bool> ink_columns(
      static_cast<std::size_t>(features.active_width), false);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < features.active_width; ++x) {
      const float darkness =
          std::clamp((1.0f - grayscale_at(nchw, height, width, y, x)) * 0.5f,
                     0.0f, 1.0f);
      if (darkness < kInkDarknessMin) {
        continue;
      }
      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
      ink_columns[static_cast<std::size_t>(x)] = true;
      ++features.ink_pixels;
    }
  }
  if (features.ink_pixels == 0) {
    return features;
  }

  features.bbox_width = max_x - min_x + 1;
  features.bbox_height = max_y - min_y + 1;
  const int ink_column_count = static_cast<int>(std::count(
      ink_columns.begin() + min_x, ink_columns.begin() + max_x + 1, true));
  features.width_ratio =
      static_cast<float>(features.bbox_width) /
      static_cast<float>(features.active_width);
  features.aspect_ratio =
      static_cast<float>(features.bbox_width) /
      static_cast<float>(features.bbox_height);
  features.height_ratio =
      static_cast<float>(features.bbox_height) / static_cast<float>(height);
  features.ink_column_coverage =
      static_cast<float>(ink_column_count) /
      static_cast<float>(features.bbox_width);
  features.matched =
      features.bbox_width >= kMinBboxWidth &&
      features.width_ratio >= kMinWidthRatio &&
      features.aspect_ratio >= kMinAspectRatio &&
      features.height_ratio <= kMaxHeightRatio &&
      features.ink_column_coverage >= kMinInkColumnCoverage;
  return features;
}

} // namespace ppocrv6_native::recognition
