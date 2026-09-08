# Native Runtime

This directory contains the C++/CUDA/TensorRT runtime and C ABI used by the
Rust server.

Native responsibilities:

- Shared TensorRT recognizer engine/classifier lifetime with per-worker
  execution contexts and a shared per-device enqueue workspace.
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

## Enqueue workspace ownership

Detector and recognition contexts share one CUDA allocation, sized at worker
construction to the largest engine requirement. Weights, context-persistent
state, I/O, and classifier buffers remain separate. A native mutex lease covers
profile selection, enqueue, and stream completion; exception unwinding also
drains the stream before releasing ownership. A failed drain poisons the arena
and rejects subsequent leases. This is independent of HTTP worker permits and
also protects direct native callers. Concurrent CPU work remains possible, but
TensorRT stages using the same arena are serialized.

Each enqueue rebinds the current workspace address, so constructing a worker
with a larger requirement cannot leave a stale address in use. Arena growth
waits for outstanding leases and preserves the old allocation if allocation
fails. Weak registry entries do not retain GPU memory after workers are gone.
Workers must execute on their construction CUDA device, as with their other
device allocations.

`PPOCRV6_SHARED_TRT_WORKSPACE=0` selects independent per-worker arenas for
deployments that prioritize concurrent GPU execution. Set this before creating
workers and keep it fixed for the process lifetime. It does not restore the old
lazy profile allocation: both modes reserve the engine requirement at startup.

`/v1/ocr/info` and glyph info expose `workspace_shared` and
`workspace_reserved_bytes`. Shared bytes refer to the same arena and must not
be summed across contexts. `context_memory_bytes` reports the detector's
requirement or the recognizer's last selected profile requirement (zero until
recognition shape selection requests it), not an additional owned allocation.

Host-input staging buffers are allocated only for host-input calls or nonzero
warmup. Full-page GPU preprocessing and ROI paths avoid these unused buffers.
Source-image storage still grows to the largest decoded page seen; budget its
configured pixel limit separately (about 46 MiB at 16 million RGB pixels).

Build/run `ppocrv6_trt_workspace_smoke` for ownership, growth, concurrent writes,
exception-drain, and lifetime checks. `tools/verify_workspace_http.py` compares
semantic response hashes against an isolated baseline, exercises mixed HTTP
requests, and records memory and timings using explicitly supplied local
fixtures. It requires Pillow, preserves no recognized text in reports, and is
a regression comparison rather than an independent accuracy benchmark.
