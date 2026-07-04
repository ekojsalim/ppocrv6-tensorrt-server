# PP-OCRv6 TensorRT Server

High-performance PP-OCRv6 inference server with a Rust HTTP/API layer and a
native C++/CUDA/TensorRT runtime.

## Features

- Rust HTTP server with health, glyph recognition, and full-page OCR endpoints.
- Request limits, image limits, worker permits, and queue timeout controls at
  the API layer.
- Coarse C ABI boundary: one native call per glyph batch or page request, with
  no CUDA streams, TensorRT contexts, or device pointers exposed to Rust.
- TensorRT DB detector and TensorRT hidden recognizer workers with long-lived
  engine/context state.
- Multi-profile recognizer engine for both cropped glyph batches and full-line
  OCR crops.
- CUDA ROI warp and recognition preprocessing for detected text boxes.
- Custom CUDA classifier and native CTC decode. The recognizer is cut before
  the final PP-OCRv6 classifier, so native code applies the FP16 classifier and
  emits decoded text without materializing the large `[N,T,18710]` logits
  tensor, reducing classifier-stage VRAM pressure.
- Public model preparation helpers for downloading PP-OCRv6 ONNX files,
  deriving the hidden recognizer, and exporting classifier weights.
- Dependency-light generated examples and an end-to-end smoke test.
- TensorRT/CUDA/OpenCV/Clipper container build recipe for reproducible local
  builds.

The production path is:

```text
Rust HTTP server
  -> base64/image decode, request limits, admission, queue timeout
  -> coarse C ABI call
  -> C++/CUDA/TensorRT full-page worker
       -> OpenCV detector preprocessing
       -> TensorRT DB detector
       -> OpenCV/Clipper DB postprocess
       -> CUDA ROI warp + recognition preprocessing
       -> TensorRT hidden recognizer
       -> CUDA classifier + CTC decode
```

The native recognizer cuts the PP-OCRv6 recognition network before the final
classifier and avoids materializing the large `[N,T,18710]` logits tensor.

## Endpoints

The server currently provides:

- `GET /health` and `GET /healthz`
- `GET /v1/glyphs/info`
- `POST /v1/glyphs/recognize`
- `GET /v1/ocr/info`
- `POST /v1/ocr/recognize`
- `POST /v1/pages/recognize`
- native C ABI for long-lived recognition and full-page OCR workers
- Docker build base with TensorRT, CUDA, OpenCV, and Clipper dependencies

## Artifact Policy

Model weights, TensorRT engines, classifier raw files, generated samples, and
local build outputs are not committed. See
[docs/model-artifacts.md](docs/model-artifacts.md).

## Layout

```text
native/               C++/CUDA/TensorRT runtime and C ABI
server/               Rust HTTP server and native bindings
docker/               TensorRT/OpenCV build image
tools/                model artifact preparation and HTTP benchmark helpers
docs/                 API, build, artifact, and architecture notes
artifacts/            ignored local model artifact staging directory
examples/             generated sample images and request helpers
```

## Quick Start

For a complete end-to-end validation checklist, use
[docs/smoke-test.md](docs/smoke-test.md).

Build the helper image:

```bash
podman build \
  -f docker/tensorrt-opencv.Containerfile \
  -t localhost/ppocrv6-tensorrt-server:dev \
  .
```

Build native code inside the TensorRT/OpenCV image:

```bash
podman run --rm --entrypoint bash \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  localhost/ppocrv6-tensorrt-server:dev \
  -lc 'cmake -S native -B native/build-trt-opencv &&
       cmake --build native/build-trt-opencv -j2 --target ppocrv6_native_c_abi'
```

CMake detects visible NVIDIA GPU architectures with `nvidia-smi`. For release
artifacts or cross-builds, pass an explicit value such as
`-DCMAKE_CUDA_ARCHITECTURES=89`.

Build the Rust server:

```bash
cargo build --release --manifest-path server/Cargo.toml
```

Prepare the model/classifier artifacts and TensorRT engines before starting the
server. The default local layout is under `artifacts/ppocrv6-medium/`; see
[docs/model-artifacts.md](docs/model-artifacts.md) and
[docs/smoke-test.md](docs/smoke-test.md) for the full engine
build path.

Run on the default smoke-test port `8184`:

```bash
podman run --rm --device nvidia.com/gpu=all --network host --entrypoint bash \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  localhost/ppocrv6-tensorrt-server:dev \
  -lc 'exec server/target/release/ppocrv6-tensorrt-server \
        --native-lib native/build-trt-opencv/libppocrv6_native_c_abi.so \
        --host 0.0.0.0 --port 8184 \
        --enable-ocr'
```

## Docs

- [docs/api.md](docs/api.md) - HTTP contract.
- [docs/smoke-test.md](docs/smoke-test.md) - end-to-end smoke test.
- [docs/build.md](docs/build.md) - build and run commands.
- [docs/model-artifacts.md](docs/model-artifacts.md) - required model files.
- [docs/architecture.md](docs/architecture.md) - runtime boundary and portability notes.

## License

MIT. See [LICENSE](LICENSE).
