# Detector minimum-shape fix

Built and promoted locally on 2026-09-06. Previously, valid small or extreme
aspect-ratio images reached TensorRT with a detector dimension below 256,
returning HTTP 500. The default longest-side limit of 1280 also caused wide
or tall source images to acquire an unsupported short side during resizing.

## Implementation

The existing resize and multiple-of-32 rounding remain unchanged. The detection
worker queries the loaded engine's actual profile bounds and chooses the
smallest 32-aligned canvas that contains the resized image, fits a batch-one
RGB input, and respects configured maximum dimensions. Unsupported deployment
configurations still fail explicitly rather than silently exceeding buffers.

CUDA preprocessing resizes into the upper-left content rectangle and fills
right/bottom padding with normalized white. Existing unpadded requests use the
same interpolation and normalization. No additional GPU buffers or workspace
are allocated.

Before DB contour extraction, scoring, unclipping, or source-coordinate scaling,
the host detector map is compacted in place to remove padded rows/columns.
The code verifies that content boundaries map exactly onto output-map pixels.
This excludes padding detections and lets existing postprocessing map content
coordinates back to the original image. Reported detector tensor/output shapes
continue to describe the actual padded engine tensors. The native API accepting
an already-preprocessed detector tensor retains its previous behavior.

This is padding, not a new upscaling or tiling strategy. Existing 32-pixel resize
rounding and recognition-width limits remain in effect; successful acceptance
of extremely thin inputs is not a claim of useful recognition accuracy on them.

## Validation with vLLM resident

- Rebuilt the native shared library successfully in the retained TensorRT 10.14
  builder and layered it onto the previously promoted Rust/runtime image.
- `tools/verify_detector_shapes_http.py`: 18 blank shapes passed, including
  640×192, 640×239, 128×128, 2560×256, 256×2560, 3200×480,
  1×4000, 4000×1, and 4000×4000. Blank padding produced no text boxes.
- Five text-bearing small/wide/tall cases matched explicitly white-padded
  control images exactly in text and original-source box coordinates. The
  cases include two inputs requiring 2× coordinate rescaling after detection.
- Page/glyph regression: all 126 cases passed expected statuses, plus 50 mixed
  concurrent requests without mismatches. Against the promoted pre-fix engine,
  125 case hashes/statuses are unchanged; only `blank-192x640` changes from
  HTTP 500 to HTTP 200. Its test expectation now requires success.
- Line suite: 100 retained crops, 14 boundaries, 10 invalid-input controls,
  and 30 concurrent requests passed.
- Whole-device memory after page/glyph tests: 31,893 MiB; after line staging:
  31,915 MiB, with 236 MiB free, matching the pre-fix warmed configuration.
  No OOM occurred. vLLM was not restarted or reconfigured.

These fixtures check geometry and regressions; they are not a broad labeled
production accuracy evaluation.

## Selected image and reproduction

`localhost/ppocrv6-vllm-ocr:dev` and `:detector-padding-v1` reference:
`841db5105303b2278dfa7725ecd9792824d9f9cb95dcfddaa923842c0d8dbaf7`.
Native library SHA-256:
`fa3ed8b70cdc16bff61c038bbd2a2fdbf9f4476224e0a5a6474bb7de57aa94e7`.
The mounted detector and accepted recognition engine are unchanged; no TensorRT
engine rebuild was required. The existing Rust binary is also unchanged.

Artifacts and build recipe: `tmp/detector-padding-20260906/`.
Run the focused test against the running orchestrator:

```bash
uv run --no-project --python /usr/bin/python3 python tools/verify_detector_shapes_http.py \
  --out tmp/detector-padding-20260906/shapes.json
```

The pre-fix image is retained as `:rollback-pre-padding` (also
`:line-preprocess-v2`). To roll back this server-only fix, stop OCR, retag that
image as `:dev`, and start OCR again through the orchestrator. The mounted
engine does not need changing for this rollback.
