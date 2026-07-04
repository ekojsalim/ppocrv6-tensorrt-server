#pragma once

#include "ppocrv6_native/common/cuda_check.h"

#include <cstddef>
#include <utility>

namespace ppocrv6_native {

template <typename T>
class CudaPtr {
public:
  CudaPtr() = default;

  explicit CudaPtr(std::size_t count) { reset(count); }

  ~CudaPtr() noexcept {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  CudaPtr(CudaPtr &&other) noexcept : ptr_(std::exchange(other.ptr_, nullptr)) {}

  CudaPtr &operator=(CudaPtr &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cudaFree(ptr_);
      }
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }

  CudaPtr(const CudaPtr &) = delete;
  CudaPtr &operator=(const CudaPtr &) = delete;

  [[nodiscard]] T *get() noexcept { return ptr_; }
  [[nodiscard]] const T *get() const noexcept { return ptr_; }
  [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

  void reset(std::size_t count) {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    if (count > 0) {
      PPOCRV6_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
  }

  [[nodiscard]] T *release() noexcept { return std::exchange(ptr_, nullptr); }

private:
  T *ptr_ = nullptr;
};

template <typename T>
class CudaHostPtr {
public:
  CudaHostPtr() = default;

  explicit CudaHostPtr(std::size_t count) { reset(count); }

  ~CudaHostPtr() noexcept {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
    }
  }

  CudaHostPtr(CudaHostPtr &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)) {}

  CudaHostPtr &operator=(CudaHostPtr &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cudaFreeHost(ptr_);
      }
      ptr_ = std::exchange(other.ptr_, nullptr);
    }
    return *this;
  }

  CudaHostPtr(const CudaHostPtr &) = delete;
  CudaHostPtr &operator=(const CudaHostPtr &) = delete;

  [[nodiscard]] T *get() noexcept { return ptr_; }
  [[nodiscard]] const T *get() const noexcept { return ptr_; }
  [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

  void reset(std::size_t count) {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
    }
    if (count > 0) {
      PPOCRV6_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    }
  }

  [[nodiscard]] T *release() noexcept { return std::exchange(ptr_, nullptr); }

private:
  T *ptr_ = nullptr;
};

} // namespace ppocrv6_native
