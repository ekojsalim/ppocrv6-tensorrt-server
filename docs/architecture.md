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
short lines to the smallest supported line width instead of failing. Contexts
share a per-device enqueue workspace reserved at worker construction for the
largest engine requirement. Native leases serialize access through CUDA stream
completion, including exception cleanup; the HTTP permit count is not relied
on for workspace safety. Set `PPOCRV6_SHARED_TRT_WORKSPACE=0` before startup for
independent per-worker workspaces when concurrent GPU execution is preferred.
Host-input buffers are allocated only for host-input calls or warmup, so the
full-page GPU input paths avoid duplicate staging allocations. See
[native ownership details](../native/README.md#enqueue-workspace-ownership).

## Cropped Lines and Shared Recognition

The Rust line route decodes cropped upright lines, checks aggregate byte/pixel
budgets before inference, selects engine-supported width buckets, and creates
one normalized host tensor chunk at a time. It uses height 48 and proportional
width with normalized-zero right padding; it never uses glyph containment to
squeeze an over-wide line. Results are restored to input order.
Multi-image chunks with at least 131,072 float elements resize independent
images in parallel through the existing Rayon pool. Smaller chunks stay serial
to avoid scheduling overhead. Each task writes a disjoint tensor slice using
the same interpolation arithmetic; this changes neither the input tensor's
layout nor its allocation bound.

`RecognitionWorker` is the common recognition component. A shared ownership
constructor lets `FullPageWorker` use the same instance that the line route
calls. Line requests use its host-tensor entry point; full-page ROI processing
uses its device-tensor entry point. Its native mutex protects mutable context
and classifier buffers even when the two Rust routes run concurrently.
The native full-page worker retains ownership if the original recognizer handle
is destroyed first. The existing standalone full-page C ABI remains available.

The server creates this line worker when either lines or full OCR are enabled.
With only `--enable-lines`, detector construction is skipped. Glyphs retain their
separate worker and policies, sharing immutable engine resources and the common
enqueue arena as before. Native library and Rust server must be deployed
together because the new server uses the shared-worker C ABI symbols.

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
