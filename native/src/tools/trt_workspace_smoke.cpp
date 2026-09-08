#include "ppocrv6_native/common/trt_workspace.h"

#include <future>
#include <iostream>
#include <vector>

using namespace ppocrv6_native;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  constexpr std::size_t bytes = 4 * 1024 * 1024;
  auto a = acquire_trt_workspace(bytes);
  auto b = acquire_trt_workspace(bytes * 2);
  require((a == b) == shared_trt_workspace_enabled(), "workspace ownership");
  require(b->size() >= bytes * 2, "workspace growth");
  // Exercise actual overlap independent of the HTTP server's permit count.
  auto run = [&a](unsigned char value) {
    cudaStream_t stream;
    PPOCRV6_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<unsigned char> host(bytes);
    for (int i = 0; i < 100; ++i) {
      TrtWorkspace::Lease lease(*a, stream);
      PPOCRV6_CUDA_CHECK(cudaMemsetAsync(lease.data(), value, bytes, stream));
      PPOCRV6_CUDA_CHECK(cudaMemcpyAsync(host.data(), lease.data(), bytes,
                                        cudaMemcpyDeviceToHost, stream));
      PPOCRV6_CUDA_CHECK(cudaStreamSynchronize(stream));
      for (auto byte : host) require(byte == value, "overlapping workspace writes");
    }
    // A host exception after CUDA submission must also drain before unlock.
    struct InjectedFailure {};
    try {
      TrtWorkspace::Lease lease(*a, stream);
      PPOCRV6_CUDA_CHECK(cudaMemsetAsync(lease.data(), value, bytes, stream));
      throw InjectedFailure{};
    } catch (const InjectedFailure &) {
      PPOCRV6_CUDA_CHECK(cudaStreamQuery(stream));
    }
    PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));
  };
  auto first = std::async(std::launch::async, run, 0x3c);
  auto second = std::async(std::launch::async, run, 0xa5);
  first.get();
  second.get();
  std::weak_ptr<TrtWorkspace> lifetime = b;
  a.reset();
  b.reset();
  require(lifetime.expired(), "workspace lifetime");
  std::cout << "PASS workspace growth, ownership, concurrent writes, exception drain\n";
}
