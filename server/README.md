# Rust Server

The Rust crate owns the HTTP/API surface and keeps the native runtime behind a
coarse C ABI. It intentionally does not expose CUDA streams, TensorRT contexts,
or device pointers across FFI.

Endpoints:

```text
GET  /health
GET  /healthz
GET  /v1/glyphs/info
POST /v1/glyphs/recognize
GET  /v1/ocr/info
POST /v1/ocr/recognize
POST /v1/pages/recognize
```

Build:

```bash
cargo build --release --manifest-path server/Cargo.toml
```

Run inside a TensorRT runtime image so `libnvinfer`, `libcudart`, OpenCV, and
Clipper resolve:

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

The glyph and full-page OCR endpoints share the same recognizer artifacts.
Full-page OCR is enabled only with `--enable-ocr`.

Glyph requests use `cjk_focus_fallback` by default, masking ASCII and visually
confusable straight-line punctuation and conservatively recovering strong CJK
candidates when the normal glyph decode is empty. Select `cjk_focus` to retain
the masking without empty-result recovery, or `suppress_ascii` for the narrower
legacy policy. Set
`"character_policy": "all"` per request, or
`PPOCRV6_GLYPH_CHARACTER_POLICY=all` for a server-wide default, when glyph
traffic intentionally contains ASCII. Full-page OCR remains unrestricted.

See [../docs/api.md](../docs/api.md) for request/response details.
