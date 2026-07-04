#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ppocrv6_native {

inline void cuda_check(cudaError_t result, const char *expr, const char *file,
                       int line) {
  if (result == cudaSuccess) {
    return;
  }
  throw std::runtime_error(std::string("CUDA error at ") + file + ":" +
                           std::to_string(line) + " while running `" + expr +
                           "`: " + cudaGetErrorString(result));
}

} // namespace ppocrv6_native

#define PPOCRV6_CUDA_CHECK(expr)                                               \
  ::ppocrv6_native::cuda_check((expr), #expr, __FILE__, __LINE__)
