#include "ppocrv6_native/recognition/ctc_decode.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using ppocrv6_native::recognition::ctc_greedy_decode_batch;

namespace {

void expect(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  const std::vector<std::string> characters = {"blank", "a", "b", "c", " "};
  const std::vector<int> ids = {
      0, 1, 1, 0, 2, 2, 3, 0,
      2, 0, 2, 2, 0, 1, 1, 3,
  };
  const std::vector<float> scores = {
      1.0f, 0.8f, 0.7f, 1.0f, 0.7f, 0.6f, 0.9f, 1.0f,
      0.5f, 1.0f, 0.6f, 0.4f, 1.0f, 0.9f, 0.8f, 0.7f,
  };

  const auto decoded = ctc_greedy_decode_batch(ids, scores, 2, 8, characters);
  expect(decoded.size() == 2, "decoded batch size mismatch");
  expect(decoded[0].text == "abc", "row 0 decoded text mismatch");
  expect(std::fabs(decoded[0].score - 0.8f) < 1e-6f,
         "row 0 decoded score mismatch");
  expect(decoded[1].text == "bbac", "row 1 decoded text mismatch");
  expect(std::fabs(decoded[1].score - 0.675f) < 1e-6f,
         "row 1 decoded score mismatch");

  std::cout << "ctc_smoke: ok rows=" << decoded.size() << '\n';
  return 0;
}
