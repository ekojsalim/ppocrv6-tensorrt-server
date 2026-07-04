#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace ppocrv6_native {

// Compute an inverse perspective transform mapping destination quad points to
// source quad points. Points are flattened as x0,y0,x1,y1,x2,y2,x3,y3.
[[nodiscard]] inline std::array<float, 9>
compute_perspective_inv(const std::array<float, 8> &dst_pts,
                        const std::array<float, 8> &src_pts) {
  double x0 = dst_pts[0], y0 = dst_pts[1], x1 = dst_pts[2],
         y1 = dst_pts[3];
  double x2 = dst_pts[4], y2 = dst_pts[5], x3 = dst_pts[6],
         y3 = dst_pts[7];
  double u0 = src_pts[0], v0 = src_pts[1], u1 = src_pts[2],
         v1 = src_pts[3];
  double u2 = src_pts[4], v2 = src_pts[5], u3 = src_pts[6],
         v3 = src_pts[7];

  double a[8][8] = {
      {x0, y0, 1.0, 0.0, 0.0, 0.0, -x0 * u0, -y0 * u0},
      {x1, y1, 1.0, 0.0, 0.0, 0.0, -x1 * u1, -y1 * u1},
      {x2, y2, 1.0, 0.0, 0.0, 0.0, -x2 * u2, -y2 * u2},
      {x3, y3, 1.0, 0.0, 0.0, 0.0, -x3 * u3, -y3 * u3},
      {0.0, 0.0, 0.0, x0, y0, 1.0, -x0 * v0, -y0 * v0},
      {0.0, 0.0, 0.0, x1, y1, 1.0, -x1 * v1, -y1 * v1},
      {0.0, 0.0, 0.0, x2, y2, 1.0, -x2 * v2, -y2 * v2},
      {0.0, 0.0, 0.0, x3, y3, 1.0, -x3 * v3, -y3 * v3},
  };
  double b[8] = {u0, u1, u2, u3, v0, v1, v2, v3};

  for (int col = 0; col < 8; ++col) {
    int pivot = col;
    double max_abs = std::abs(a[col][col]);
    for (int row = col + 1; row < 8; ++row) {
      const double candidate = std::abs(a[row][col]);
      if (candidate > max_abs) {
        max_abs = candidate;
        pivot = row;
      }
    }
    if (pivot != col) {
      std::swap(b[col], b[pivot]);
      for (int k = 0; k < 8; ++k) {
        std::swap(a[col][k], a[pivot][k]);
      }
    }

    double denom = a[col][col];
    if (std::abs(denom) < 1e-12) {
      denom = (denom < 0.0) ? -1e-12 : 1e-12;
    }
    for (int k = col; k < 8; ++k) {
      a[col][k] /= denom;
    }
    b[col] /= denom;

    for (int row = 0; row < 8; ++row) {
      if (row == col) {
        continue;
      }
      const double factor = a[row][col];
      for (int k = col; k < 8; ++k) {
        a[row][k] -= factor * a[col][k];
      }
      b[row] -= factor * b[col];
    }
  }

  return {static_cast<float>(b[0]), static_cast<float>(b[1]),
          static_cast<float>(b[2]), static_cast<float>(b[3]),
          static_cast<float>(b[4]), static_cast<float>(b[5]),
          static_cast<float>(b[6]), static_cast<float>(b[7]), 1.0f};
}

} // namespace ppocrv6_native
