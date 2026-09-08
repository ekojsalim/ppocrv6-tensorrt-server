# API Contract

Default local test URL:

```text
http://127.0.0.1:8184
```

The Rust server owns HTTP admission, queueing, request limits, base64/image
decode, and timeouts. Native C++/CUDA/TensorRT owns detector preprocessing,
detection, DB postprocess, ROI warp, recognition, classifier, and CTC decode.

Full-page OCR accepts small and extreme-aspect-ratio images within its normal
byte/pixel limits. Detector preprocessing retains the configured resize and
32-pixel rounding, then pads the right/bottom with white to fit the smallest
compatible engine profile. Padding is excluded before text-box extraction;
returned boxes remain in original-image coordinates. With the deployed engine,
a 640×192 image uses a 640×256 detector canvas without stretching its content.

## Health

```http
GET /health
GET /healthz
```

Response:

```json
{"ok": true}
```

## Full-Page OCR

For already-cropped text lines, use the line endpoint below to skip detection.

```http
GET /v1/ocr/info
POST /v1/ocr/recognize
POST /v1/pages/recognize
```

`/v1/ocr/info` returns native detector/recognizer metadata, OCR request limits,
and detector preprocess config. For the production-shaped path, native input is
decoded RGB bytes and C++ owns CUDA detector resize/color conversion/normalize.
OpenCV/Clipper remains responsible for DB postprocessing.

Recognizer metadata includes `shared_engine_use_count`,
`context_memory_bytes`, and `profile_memory_bytes`. The first identifies shared
immutable engine/classifier use; the latter two expose the last selected
profile's requirement (zero before selection) and each profile's requirement.
`workspace_shared` identifies shared enqueue-memory ownership and
`workspace_reserved_bytes` reports the arena allocation. When shared, that
allocation is common to detector and recognizer contexts: do not sum it across
their metadata records. Engine weights, persistent context state, and other
buffers are additional to this arena.

Request body for `/v1/ocr/recognize`:

```json
{
  "image": "<base64 PNG/JPEG/WebP/etc, data URI also accepted>"
}
```

Minimal Python request:

```python
import base64, json, urllib.request
from pathlib import Path

payload = json.dumps({
    "image": base64.b64encode(Path("page.png").read_bytes()).decode("ascii")
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8184/v1/ocr/recognize",
    data=payload,
    headers={"Content-Type": "application/json"},
)
result = json.load(urllib.request.urlopen(req, timeout=120))
```

Important response fields:

```json
{
  "count": 98,
  "source_shape": [1584, 1224],
  "detector_input_shape": [1, 3, 1280, 992],
  "timing": {
    "total_ms": 57.7,
    "detector_preprocess_ms": 12.5,
    "detect_ms": 6.5,
    "postprocess_ms": 1.9,
    "roi_ms": 0.2,
    "recognize_ms": 35.6
  },
  "lines": [
    {
      "index": 0,
      "text": "...",
      "score": 0.99,
      "det_score": 0.92,
      "box": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    }
  ],
  "rust_timings": {
    "rust_preprocess_ms": 3.6,
    "rust_base64_decode_ms": 0.2,
    "rust_image_decode_ms": 3.4,
    "rust_blocking_ms": 63.3
  }
}
```

## Cropped Line Recognition

```http
GET /v1/lines/info
POST /v1/lines/recognize
```

Enabled by `--enable-ocr`, or independently by `--enable-lines` /
`PPOCRV6_ENABLE_LINES=true`. The latter loads no detector. Glyph recognition
remains available in both modes. When lines are disabled these routes return 503.

Supply already-cropped, upright **single text lines** (one or multiple
characters). This endpoint does not find text boxes, split paragraphs, or rotate
vertical text. It always uses unrestricted characters and model confidence;
glyph suppression and single-stroke fallback do not apply.

```json
{"images": ["<base64 line crop>", "<base64 line crop>"], "batch_size": 12}
```

Supply exactly one of `image` or nonempty `images`. PNG, JPEG, WebP, BMP, PNM,
and data-URI inputs use the existing image decoder. Images resize proportionally
to height 48 and are right-padded with normalized zero to the smallest supported
width bucket. The current engine accepts buckets 128, 640, 960, 1280, 1600, 2400,
3200. Width selection follows the loaded engine; clients should not hardcode it.
A crop requiring `ceil(48 * width / height) > 3200` currently returns 400;
split it into shorter lines instead of relying on silent clipping.

Request image count and inference batch size are independent. The defaults are:

| Limit | Default | Configuration |
| --- | ---: | --- |
| Images per request | 128 | `--lines-max-images` / `PPOCRV6_LINES_MAX_IMAGES` |
| Total decoded image-file bytes | 16 MiB | `--lines-max-total-bytes` / `PPOCRV6_LINES_MAX_TOTAL_BYTES` |
| Total source pixels | 16,000,000 | `--lines-max-total-pixels` / `PPOCRV6_LINES_MAX_TOTAL_PIXELS` |
| Inference chunk | 12 | Optional request `batch_size`, 1 through configured maximum |

The byte limit counts decoded base64 bytes, before image decompression, summed
across all inputs including duplicates. Pixel limits sum source dimensions;
headers are checked before pixel allocation. The HTTP JSON body is also bounded.
All input validation completes before inference. Limits return 413, invalid
images/geometry or conflicting fields return 400, and schema/type errors or
unknown request fields return 422. `width`, `character_policy`, and `score_mode`
are not line request options. A batch fails as a whole; no partial successes are
returned. The shared server permit and queue-timeout settings also apply.

```json
{
  "count": 2,
  "character_policy": "all",
  "score_mode": "model",
  "predictions": [
    {"index": 0, "text": "First line", "score": 0.98, "bucket_width": 640},
    {"index": 1, "text": "Second line", "score": 0.96, "bucket_width": 640}
  ],
  "recognition_chunks": [{"batch": 2, "width": 640}],
  "timing": {"decode_ms": 1.0, "total_ms": 3.0}
}
```

Predictions always retain input order even though inference groups similar
widths. Each also includes `class_ids` and `per_char_scores`. Single `image`
requests additionally return `prediction`, an alias for `predictions[0]`.
There are no detector boxes or detection scores.

Line timing includes `decode_ms` (decode and bucket selection), `resize_ms`
(chunk allocation and resize/normalization), and `recognize_ms` (native calls
including result conversion). `native_predict_ms` is a subset of `recognize_ms`
reported by the native worker, not a separate additive stage or a GPU-only
measurement. `total_ms` covers the blocking handler, excluding HTTP admission
queue time. Larger multi-image chunks resize in parallel using the existing
Rayon pool; normalized tensor storage remains bounded to one inference chunk.

`/v1/lines/info` publishes effective buckets, normalized-width and request
limits, and native recognizer memory metadata. It shares the **same recognition
worker/context/classifier buffers** with full-page OCR. Existing `--ocr-rec-*`
settings configure this common worker in either mode. GPU host-input staging
is allocated on first line use (about 22 MiB at current limits); it does not
create another recognition context or enqueue workspace.

## Glyph Recognition

```http
GET /v1/glyphs/info
POST /v1/glyphs/recognize
```

Request body:

```json
{
  "image": "<base64 image>"
}
```

or:

```json
{
  "images": ["<base64 image>", "<base64 image>"],
  "width": 80,
  "batch_size": 256,
  "return_timesteps": false,
  "character_policy": "cjk_focus_fallback",
  "score_mode": "accepted"
}
```

`image` returns a convenience `prediction` field in addition to `predictions`.
`images` returns one item per input in `predictions`.
The defaults accept up to 1,024 images in one logical request and execute them
in TensorRT chunks of up to 256 images.

Glyph preprocessing preserves the source aspect ratio inside the requested
`width` by 48-pixel recognition tensor. Inputs whose height-normalized width
fits are unchanged (for example, 48x48 and 53x48 at the default width of 80).
Wider inputs are scaled down on both axes and centered vertically with white
padding instead of being squeezed horizontally.

Glyph recognition defaults to `"character_policy": "cjk_focus_fallback"`.
Its primary pass uses `cjk_focus`, masking ASCII classifier classes plus
straight-line tokens visually confusable with CJK strokes: macron `¯`, en dash
`–`, em dash `—`, horizontal bar `―`, minus sign `−`, horizontal-line
extension `⎯`, and fullwidth hyphen-minus `－`. Masking happens before
softmax/argmax and CTC decode, reducing failures such as `一` becoming `_`,
`-`, or `–`, and `入` becoming `T`.

Set `"character_policy": "suppress_ascii"` for the narrower legacy policy or
`"character_policy": "all"` to opt out for requests that legitimately contain
ASCII or the masked line symbols. Character policies are glyph-only; full-page
OCR always uses the unrestricted vocabulary.

When normal `cjk_focus` CTC decoding is empty, the default policy checks only
the common single-stroke character `一`; it does not search for or synthesize
any other CJK character. A small CUDA classifier evaluates the existing
PP-OCRv6 `一` classifier column against the already-known blank logit. The
result is accepted only when that one-vs-blank probability is at least `0.005`
and the already-preprocessed pixels form one long, thin, horizontally coherent
stroke. The shape gate keeps blank, dot, vertical, diagonal, box, and multi-line
controls out of the fallback.

This path is substantially cheaper than a second full-vocabulary classifier
pass. Responses include `empty_fallback_attempted_count`,
`empty_fallback_applied_count`, `one_stroke_fallback_applied_count`, and
per-prediction `empty_fallback` diagnostics. A recovery reports
`empty_fallback.applied_by: "one_stroke"` and
`score_type: "one_vs_blank_probability"`. Set
`"character_policy": "cjk_focus"` to disable the `一` fallback while retaining
the same vocabulary masking.

Suppression does not rewrite punctuation into a guessed CJK character. It lets
the classifier choose among blank and the remaining classes.

Glyph recognition defaults to `"score_mode": "accepted"`. Any non-empty result
accepted by the character/fallback policy returns `score: 1.0` and
`per_char_scores` of `1.0`; an empty result remains `0.0`. Its `score_type` is
`binary_acceptance`, not a calibrated probability. This prevents generic
client-side probability thresholds from discarding low-confidence but
policy-approved glyphs.

Accepted mode is also a classifier optimization: normal chunks skip the
vocabulary-wide exponential/sum reduction and probability device-to-host copy.
If a `cjk_focus_fallback` primary decode is empty, the server calculates only
the dedicated `一`-vs-blank probability used by the shape-gated fallback. That
value remains visible in the per-prediction `empty_fallback` diagnostics.

Set `"score_mode": "model"` when the caller needs model confidence. In that
mode, `cjk_focus`, `suppress_ascii`, and the primary pass of
`cjk_focus_fallback` use `score_type: "conditional_probability"` because scores
are normalized over the allowed vocabulary; `"all"` uses
`score_type: "probability"`. `model_score_type` reports which of those
probability interpretations applies even when accepted mode is active.

`character_policy`, `score_mode`, `score_type`, and `model_score_type` are
returned at the top level and in `meta`. `GET /v1/glyphs/info` reports
`default_character_policy`, `default_score_mode`, and
`supported_score_modes`.

## Errors

Errors use HTTP status codes and a JSON body:

```json
{
  "error": "message",
  "status": 400
}
```

Common statuses:

```text
400 invalid JSON/base64/image/options
413 request or image exceeds configured limits
503 worker unavailable or queue timeout
500 native/runtime failure
```
