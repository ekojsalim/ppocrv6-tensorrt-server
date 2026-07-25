# Architecture

The server is split at a coarse FFI boundary:

```text
Rust
  HTTP, JSON, base64/image decode
  request limits, admission, timeout/backpressure
  one native call per glyph batch or page request

C++/CUDA/TensorRT
  shared model lifetime and independent worker contexts
  CUDA detector preprocessing
  detection and DB postprocess
  ROI warp and recognition preprocessing
  recognition bucket execution
  classifier and CTC decode
```

Rust must not receive CUDA streams, TensorRT contexts, device pointers, or
per-kernel scheduling responsibilities.

## Full-Page OCR

The full-page endpoint accepts one encoded image. Rust decodes it to RGB bytes
and calls the native full-page worker. Native code handles:

1. One RGB upload followed by CUDA resize/color conversion/normalize for the
   detector.
2. TensorRT detector inference.
3. OpenCV/Clipper DB postprocess.
4. CUDA ROI warp/normalize for each detected text crop.
5. TensorRT hidden recognizer.
6. CUDA classifier and CTC decode.

The glyph endpoint and full-page pipeline share one deserialized recognizer
engine, classifier weights, bias, and character table. They keep separate
execution contexts and streams because requests can overlap and TensorRT
context state is mutable. New recognizer engines have glyph, short-line, and
long-line profiles. The runtime filters its preferred recognition buckets
against the profiles actually present, so older glyph + long-line engines pad
short lines to the smallest supported line width instead of failing. Each
context lazily allocates workspace for the smallest-memory profile that accepts
the current shape and grows if a later request requires more.

## Glyph Recognition

The glyph endpoint accepts one image or a batch of cropped glyph images. Rust
decodes the HTTP image payloads and calls one native recognition worker batch.

## Portability

The source code is portable across modern NVIDIA GPU generations, but the built
artifacts are not all portable:

- TensorRT engines should be rebuilt or cached for the target GPU class,
  TensorRT version, precision, and profile set.
- CUDA native builds auto-detect visible NVIDIA GPU architectures with
  `nvidia-smi` and otherwise fall back to CMake's `all-major` CUDA architecture
  set. Release builds should still use an explicit architecture policy.
- TensorRT hardware compatibility modes can broaden engine compatibility at a
  possible performance cost.

Keep ONNX files and classifier-export scripts as the portable source of truth;
treat `.trt` engines as deployment cache artifacts.
