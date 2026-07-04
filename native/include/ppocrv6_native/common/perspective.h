#pragma once

#include "ppocrv6_native/common/box.h"
#include "ppocrv6_native/common/perspective_math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ppocrv6_native {

struct CropTransform {
  std::array<float, 9> m_inv{};
  int crop_width = 0;
  int natural_width = 0;
  bool vertical = false;
};

struct CanonicalBox {
  std::array<Point2f, 4> points{};
  float width = 1.0f;
  float height = 1.0f;
  bool vertical = false;
};

[[nodiscard]] inline CanonicalBox canonicalize_box_for_recognition(
    const Box &box) {
  const float width_top = distance(box[0], box[1]);
  const float width_bottom = distance(box[3], box[2]);
  const float height_left = distance(box[0], box[3]);
  const float height_right = distance(box[1], box[2]);

  float width = max_side(width_top, width_bottom);
  float height = max_side(height_left, height_right);

  CanonicalBox result{};
  result.vertical = height > width * kVerticalAspectRatio;
  if (result.vertical) {
    result.points = {box[1], box[2], box[3], box[0]};
    std::swap(width, height);
  } else {
    result.points = box.points;
  }
  result.width = width;
  result.height = height;
  return result;
}

[[nodiscard]] inline CropTransform compute_crop_transform(
    const Box &box, int target_height, int bucket_width) {
  if (target_height <= 0) {
    throw std::invalid_argument("target_height must be positive");
  }
  if (bucket_width <= 0) {
    throw std::invalid_argument("bucket_width must be positive");
  }

  const CanonicalBox canonical = canonicalize_box_for_recognition(box);
  const int natural_width = std::max(
      1, static_cast<int>(
             std::ceil(static_cast<float>(target_height) * canonical.width /
                       canonical.height)));
  const int crop_width = std::min(bucket_width, natural_width);
  const float dst_x1 = static_cast<float>(crop_width - 1);
  const float dst_y1 = static_cast<float>(target_height - 1);

  const std::array<float, 8> dst_pts = {0.0f, 0.0f, dst_x1, 0.0f,
                                        dst_x1, dst_y1, 0.0f, dst_y1};
  const std::array<float, 8> src_pts = {
      canonical.points[0].x, canonical.points[0].y,
      canonical.points[1].x, canonical.points[1].y,
      canonical.points[2].x, canonical.points[2].y,
      canonical.points[3].x, canonical.points[3].y,
  };

  CropTransform result{};
  result.m_inv = compute_perspective_inv(dst_pts, src_pts);
  result.crop_width = crop_width;
  result.natural_width = natural_width;
  result.vertical = canonical.vertical;
  return result;
}

} // namespace ppocrv6_native
