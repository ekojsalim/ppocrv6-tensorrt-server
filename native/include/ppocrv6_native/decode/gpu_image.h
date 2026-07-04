#pragma once

#include <cstddef>

namespace ppocrv6_native::decode {

struct GpuImage {
  void *data = nullptr;
  std::size_t step = 0;
  int rows = 0;
  int cols = 0;

  [[nodiscard]] constexpr bool empty() const noexcept {
    return data == nullptr || rows <= 0 || cols <= 0;
  }
};

} // namespace ppocrv6_native::decode

namespace ppocrv6_native {
using decode::GpuImage;
}
