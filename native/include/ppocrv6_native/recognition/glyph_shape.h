#pragma once

#include <span>

namespace ppocrv6_native::recognition {

struct HorizontalStrokeFeatures {
  bool matched = false;
  int active_width = 0;
  int bbox_width = 0;
  int bbox_height = 0;
  int ink_pixels = 0;
  float width_ratio = 0.0f;
  float aspect_ratio = 0.0f;
  float height_ratio = 0.0f;
  float ink_column_coverage = 0.0f;
};

HorizontalStrokeFeatures detect_long_horizontal_stroke(
    std::span<const float> nchw, int height, int width);

} // namespace ppocrv6_native::recognition
