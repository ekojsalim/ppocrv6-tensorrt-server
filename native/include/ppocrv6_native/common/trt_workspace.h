#pragma once

#include "ppocrv6_native/common/cuda_ptr.h"

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace ppocrv6_native {

// Enqueue memory, not engine weights or context-persistent state. A lease owns
// it until the CUDA stream completes, including when inference throws.
class TrtWorkspace {
public:
  class Lease {
  public:
    Lease(TrtWorkspace &owner, cudaStream_t stream)
        : owner_(owner), lock_(owner.mutex_), stream_(stream) {
      if (owner_.poisoned_) {
        throw std::runtime_error("TensorRT workspace poisoned by CUDA failure");
      }
    }
    ~Lease() noexcept {
      if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        owner_.poisoned_ = true;
      }
    }
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    void *data() const { return owner_.memory_.get(); }
    std::size_t size() const { return owner_.bytes_; }

  private:
    TrtWorkspace &owner_;
    std::unique_lock<std::mutex> lock_;
    cudaStream_t stream_;
  };

  void reserve(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (poisoned_) {
      throw std::runtime_error("TensorRT workspace poisoned by CUDA failure");
    }
    if (bytes > bytes_) {
      // Preserve the old allocation if allocation fails. No lease can be live
      // here; contexts rebind the current address for every enqueue.
      CudaPtr<unsigned char> replacement(bytes);
      memory_ = std::move(replacement);
      bytes_ = bytes;
    }
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_;
  }

private:
  mutable std::mutex mutex_;
  CudaPtr<unsigned char> memory_;
  std::size_t bytes_ = 0;
  bool poisoned_ = false;
};

inline bool shared_trt_workspace_enabled() {
  const char *value = std::getenv("PPOCRV6_SHARED_TRT_WORKSPACE");
  return value == nullptr || std::string_view(value) != "0";
}

inline std::shared_ptr<TrtWorkspace> acquire_trt_workspace(std::size_t bytes) {
  std::shared_ptr<TrtWorkspace> workspace;
  if (shared_trt_workspace_enabled()) {
    int device = 0;
    PPOCRV6_CUDA_CHECK(cudaGetDevice(&device));
    static std::mutex registry_mutex;
    static std::map<int, std::weak_ptr<TrtWorkspace>> registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    workspace = registry[device].lock();
    if (!workspace) {
      workspace = std::make_shared<TrtWorkspace>();
      registry[device] = workspace;
    }
  } else {
    workspace = std::make_shared<TrtWorkspace>();
  }
  workspace->reserve(bytes);
  return workspace;
}

} // namespace ppocrv6_native
