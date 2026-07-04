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
decoded RGB bytes and C++ owns detector resize/normalize with OpenCV.

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
  "batch_size": 64,
  "return_timesteps": false
}
```

`image` returns a convenience `prediction` field in addition to `predictions`.
`images` returns one item per input in `predictions`.

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
