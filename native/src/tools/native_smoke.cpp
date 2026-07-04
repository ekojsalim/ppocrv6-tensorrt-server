#include "ppocrv6_native/common/perspective.h"
#include "ppocrv6_native/recognition/ctc_decode.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

ppocrv6_native::Point2f map_point(const std::array<float, 9> &m, float x,
                                  float y) {
  const float denom = m[6] * x + m[7] * y + m[8];
  return {
      (m[0] * x + m[1] * y + m[2]) / denom,
      (m[3] * x + m[4] * y + m[5]) / denom,
  };
}

bool close_to(ppocrv6_native::Point2f actual, ppocrv6_native::Point2f expected,
              float tolerance = 1e-5f) {
  return std::fabs(actual.x - expected.x) <= tolerance &&
         std::fabs(actual.y - expected.y) <= tolerance;
}

} // namespace

int main() {
  const ppocrv6_native::Box box{
      {1.0f, 1.0f},
      {5.0f, 1.0f},
      {5.0f, 4.0f},
      {1.0f, 4.0f},
  };
  const auto transform = ppocrv6_native::compute_crop_transform(box, 4, 8);
  expect(transform.crop_width == 6, "unexpected crop width");
  expect(transform.natural_width == 6, "unexpected natural width");
  expect(!transform.vertical, "horizontal box was marked vertical");
  expect(close_to(map_point(transform.m_inv, 0.0f, 0.0f), {1.0f, 1.0f}),
         "homography top-left mismatch");
  expect(close_to(map_point(transform.m_inv, 5.0f, 3.0f), {5.0f, 4.0f}),
         "homography bottom-right mismatch");

  const std::vector<std::string> characters = {"blank", "x", "y"};
  const std::vector<int> ids = {0, 1, 1, 2, 0};
  const auto decoded =
      ppocrv6_native::recognition::ctc_greedy_decode(ids, {}, characters);
  expect(decoded.text == "xy", "CTC text mismatch");
  expect(std::fabs(decoded.score - 1.0f) < 1e-6f, "CTC score mismatch");

  std::cout << "native_smoke: ok crop_width=" << transform.crop_width
            << " text=" << decoded.text << '\n';
  return 0;
}
