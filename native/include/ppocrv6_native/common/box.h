#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace ppocrv6_native {

struct Point2f {
  float x = 0.0f;
  float y = 0.0f;
};

struct Box {
  std::array<Point2f, 4> points{};

  constexpr Box() = default;

  constexpr Box(Point2f p0, Point2f p1, Point2f p2, Point2f p3) noexcept
      : points{p0, p1, p2, p3} {}

  [[nodiscard]] constexpr Point2f &operator[](std::size_t index) noexcept {
    return points[index];
  }

  [[nodiscard]] constexpr const Point2f &
  operator[](std::size_t index) const noexcept {
    return points[index];
  }
};

inline constexpr float kVerticalAspectRatio = 1.5f;

[[nodiscard]] inline float distance(Point2f a, Point2f b) noexcept {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

[[nodiscard]] inline float max_side(float a, float b) noexcept {
  return std::max(std::max(a, b), 1.0f);
}

} // namespace ppocrv6_native
