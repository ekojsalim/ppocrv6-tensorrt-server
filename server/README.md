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

Glyph images are aspect-fitted into the requested width by 48-pixel tensor.
Normal near-square inputs retain their existing size; only width-overflow
inputs scale down on both axes and receive white vertical padding. This avoids
turning unusually wide single strokes into artificially thick bars.

Glyph requests use `cjk_focus_fallback` by default, masking ASCII and visually
confusable straight-line punctuation. When normal decoding is empty, it checks
only `一` with a dedicated one-vs-blank classifier plus a long-horizontal-shape
gate; it never synthesizes another CJK character from blank. Select `cjk_focus`
to retain the masking without this recovery, or `suppress_ascii` for the
narrower legacy policy. Set
`"character_policy": "all"` per request, or
`PPOCRV6_GLYPH_CHARACTER_POLICY=all` for a server-wide default, when glyph
traffic intentionally contains ASCII. Full-page OCR remains unrestricted.

The `一` fallback is inexpensive: it evaluates one classifier column against
the already-known blank logit, then runs a CPU-only shape check over the already
preprocessed glyph tensor. It does not launch the full vocabulary classifier a
second time. Responses identify recoveries with
`empty_fallback.applied_by: "one_stroke"`.

Glyph scores default to `"score_mode": "accepted"`: non-empty predictions have
`score: 1.0`, empty predictions remain `0.0`, and the native classifier skips
the probability reduction on normal chunks. Set `"score_mode": "model"` per
request, or `PPOCRV6_GLYPH_SCORE_MODE=model` server-wide, to return model
probabilities. The empty-result fallback still calculates real probabilities
internally for its acceptance gates. Full-page OCR scores are unchanged.

See [../docs/api.md](../docs/api.md) for request/response details.
