# Build And Run

These commands assume Linux with Podman, NVIDIA container support, CUDA-capable
hardware, and Rust installed on the host.

## Shared Hosts

The smoke commands below use host networking and port `8184`. On shared GPU
hosts, check local policy and current GPU/container usage before starting
long-lived or memory-heavy workloads.

## vLLM-Compatible Image

The image used by `serve-orchestrator` shares the
`docker.io/vllm/vllm-openai:v0.23.0` CUDA 13 base and uses TensorRT 10.14:

```bash
podman build \
  -f docker/vllm-ocr.Containerfile \
  -t localhost/ppocrv6-vllm-ocr:dev \
  .
```

The build installs both the optimized native library and Rust server. TensorRT
engines are runtime-version-specific, so use the TensorRT 10.14 engine bundle
in `tmp/vllm-ocr-model` with this image.

When `localhost/ppocrv6-vllm-ocr:trt10.14-native-base` already exists, use the
incremental overlay for a faster application-only rebuild:

```bash
podman build \
  -f docker/vllm-ocr-server-overlay.Containerfile \
  -t localhost/ppocrv6-vllm-ocr:dev \
  .
```

The overlay extracts only the required TensorRT development headers and links
against the shared libraries already in the base. It deliberately avoids the
roughly 2.9 GB `libnvinfer-dev` package, whose static libraries are unnecessary
for this shared-library build.

## Standalone TensorRT Image

```bash
podman build \
  -f docker/tensorrt-opencv.Containerfile \
  -t localhost/ppocrv6-tensorrt-server:dev \
  .
```

The Containerfile defaults to `nvcr.io/nvidia/tensorrt:26.03-py3`. Override
`BASE_IMAGE` at build time if you need a different TensorRT/CUDA base.

## Native Library

```bash
podman run --rm --entrypoint bash \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  localhost/ppocrv6-tensorrt-server:dev \
  -lc 'cmake -S native -B native/build-trt-opencv &&
       cmake --build native/build-trt-opencv -j2 --target ppocrv6_native_c_abi'
```

CMake detects visible NVIDIA GPU architectures with `nvidia-smi`. If no GPU is
visible during configure, it falls back to CMake's `all-major` CUDA architecture
set. For release artifacts or cross-builds, pass an explicit deployment target,
for example `-DCMAKE_CUDA_ARCHITECTURES=89`.

## Rust Server

```bash
cargo build --release --manifest-path server/Cargo.toml
```

## TensorRT Engines

Prepare model artifacts before building engines:

```bash
python3 -m pip install -e .
python3 tools/prepare_model_artifacts.py
```

Build the native engine-builder tools:

```bash
podman run --rm --entrypoint bash \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  localhost/ppocrv6-tensorrt-server:dev \
  -lc 'cmake -S native -B native/build-trt-opencv &&
       cmake --build native/build-trt-opencv -j2 \
         --target ppocrv6_trt_multi_profile_smoke ppocrv6_trt_detection_bench'
```

Build the recognizer engine:

```bash
podman run --rm --device nvidia.com/gpu=all --network host --entrypoint bash \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  localhost/ppocrv6-tensorrt-server:dev \
  -lc 'mkdir -p artifacts/ppocrv6-medium/engines &&
       native/build-trt-opencv/bin/ppocrv6_trt_multi_profile_smoke \
         --force-rebuild 1 \
         --onnx artifacts/ppocrv6-medium/derived/rec-hidden.onnx \
         --engine artifacts/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt \
         --workspace-mib 1024 \
         --glyph-min-batch 1 --glyph-opt-batch 128 --glyph-max-batch 256 \
         --glyph-min-width 48 --glyph-opt-width 80 --glyph-max-width 128 \
         --short-line-min-batch 1 --short-line-opt-batch 8 --short-line-max-batch 12 \
         --short-line-min-width 128 --short-line-opt-width 384 --short-line-max-width 640 \
         --line-min-batch 1 --line-opt-batch 8 --line-max-batch 12 \
         --line-min-width 640 --line-opt-width 1600 --line-max-width 3200'
```

Build the detector engine:

```bash
podman run --rm --device nvidia.com/gpu=all --network host --entrypoint bash \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  localhost/ppocrv6-tensorrt-server:dev \
  -lc 'mkdir -p artifacts/ppocrv6-medium/engines &&
       native/build-trt-opencv/bin/ppocrv6_trt_detection_bench \
         --force-rebuild 1 \
         --onnx artifacts/ppocrv6-medium/source/det/inference.onnx \
         --engine artifacts/ppocrv6-medium/engines/det-fp16-b1-h256-1280-w256-1280.trt \
         --fp16 1 \
         --workspace-mib 1536 \
         --min-height 256 --opt-height 1280 --max-height 1280 \
         --min-width 256 --opt-width 1280 --max-width 1280 \
         --inspect-height 992 --inspect-width 1280 \
         --warmups 0 --repeats 1 \
         --max-extra-mib 4096'
```

## Run

Run on port `8184`:

```bash
podman run --rm --device nvidia.com/gpu=all --network host --entrypoint bash \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  localhost/ppocrv6-tensorrt-server:dev \
  -lc 'exec server/target/release/ppocrv6-tensorrt-server \
        --native-lib native/build-trt-opencv/libppocrv6_native_c_abi.so \
        --host 0.0.0.0 --port 8184 \
        --enable-ocr \
        --ocr-det-engine artifacts/ppocrv6-medium/engines/det-fp16-b1-h256-1280-w256-1280.trt \
        --ocr-rec-engine artifacts/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt \
        --weight artifacts/ppocrv6-medium/classifier/weight.fp16.bin \
        --bias artifacts/ppocrv6-medium/classifier/bias.fp16.bin \
        --characters artifacts/ppocrv6-medium/classifier/characters.txt'
```

Check:

```bash
curl -s http://127.0.0.1:8184/health
curl -s http://127.0.0.1:8184/v1/ocr/info | python3 -m json.tool
```

## Glyph Benchmark

Use representative PNG or JPEG inputs for production throughput measurements.
The benchmark's generated PPM fallback is convenient for a dependency-free
smoke run, but its base64/JSON size and decode path are not production-like.
The default API accepts up to 1,024 images per logical request and executes
them in TensorRT chunks of up to 256 images. Compare TensorRT batch size
independently from HTTP request concurrency.

```bash
magick examples/samples/glyph_A.ppm /tmp/glyph_A.png

python3 tools/bench_glyph_http.py \
  --url http://127.0.0.1:8184/v1/glyphs/recognize \
  --image /tmp/glyph_A.png \
  --unique-image-payloads \
  --counts 1,6,16,32,64 \
  --width 80 \
  --batch-size 256 \
  --character-policy all
```
