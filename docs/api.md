# API Contract

Default local test URL:

```text
http://127.0.0.1:8184
```

The Rust server owns HTTP admission, queueing, request limits, base64/image
decode, and timeouts. Native C++/CUDA/TensorRT owns detector preprocessing,
detection, DB postprocess, ROI warp, recognition, classifier, and CTC decode.

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
immutable engine/classifier use; the latter two expose the context's current
allocation and each profile's possible allocation.

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
  "character_policy": "cjk_focus_fallback"
}
```

`image` returns a convenience `prediction` field in addition to `predictions`.
`images` returns one item per input in `predictions`.
The defaults accept up to 1,024 images in one logical request and execute them
in TensorRT chunks of up to 256 images.

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

When normal `cjk_focus` CTC decoding is empty, the default policy examines the
strongest CJK-ideograph alternative without suppressing CTC blank globally.
The candidate is applied only when its probability in the original
distribution is at least `0.05` and its probability among CJK-only alternatives
is at least `0.8`. Responses include `empty_fallback_attempted_count`,
`empty_fallback_applied_count`, and per-prediction `empty_fallback`
diagnostics. This policy costs an additional classifier pass only for chunks
containing an empty decode. Set `"character_policy": "cjk_focus"` to disable
only this empty-result fallback while retaining the same vocabulary masking.

Suppression does not rewrite punctuation into a guessed CJK character. It lets
the classifier choose among blank and the remaining classes. In `cjk_focus`
and `suppress_ascii` modes, and for the primary pass of
`cjk_focus_fallback`, `score_type` is `conditional_probability`, because scores
are normalized over the allowed vocabulary; with `"all"` it is `probability`.
Both `character_policy` and `score_type` are returned at the top level and in
`meta`. `GET /v1/glyphs/info` reports the configured
`default_character_policy`.

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
