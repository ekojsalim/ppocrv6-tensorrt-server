# Native Runtime

This directory contains the C++/CUDA/TensorRT runtime and C ABI used by the
Rust server.

Native responsibilities:

- Shared TensorRT recognizer engine/classifier lifetime with per-worker
  execution contexts and profile-sized workspace.
- CUDA detector resize/color conversion/normalize preprocessing.
- TensorRT DB detector execution.
- OpenCV/Clipper DB postprocess.
- CUDA ROI warp and recognition preprocessing.
- TensorRT hidden recognizer execution.
- CUDA classifier and native CTC decode.

Build with CUDA only:

```bash
cmake -S native -B native/build
cmake --build native/build -j
```

Build with TensorRT/OpenCV/Clipper:

```bash
cmake -S native -B native/build-trt-opencv
cmake --build native/build-trt-opencv -j2 --target ppocrv6_native_c_abi
```

The production build path expects OpenCV and polyclipping. If they are missing,
the detector postprocess falls back to a dependency-light smoke/dev path; do
not use that fallback for production parity.

Important targets:

```text
ppocrv6_native_c_abi
ppocrv6_full_page_worker_smoke
ppocrv6_detection_worker_smoke
ppocrv6_trt_multi_profile_smoke
ppocrv6_trt_detection_bench
```

Optional benchmark and diagnostic tools are available with:

```bash
cmake -S native -B native/build-trt-opencv \
  -DPPOCRV6_NATIVE_BUILD_BENCH_TOOLS=ON
```

CMake detects visible NVIDIA GPU architectures with `nvidia-smi`. If no GPU is
visible during configure, it falls back to CMake's `all-major` CUDA architecture
set. For release artifacts or cross-builds, pass an explicit deployment target,
for example `-DCMAKE_CUDA_ARCHITECTURES=89`.

TensorRT engines are deployment artifacts, not source artifacts. Rebuild or
cache engines for the target TensorRT version and GPU class.
