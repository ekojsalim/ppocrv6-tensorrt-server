# Cropped-line endpoint validation

Date: 2026-09-06 Asia/Jakarta. RTX 5090, existing TensorRT 10.14/CUDA 13 engines.

Implemented `GET /v1/lines/info` and `POST /v1/lines/recognize`, automatic
aspect-preserving width selection, grouping/chunking, input-order restoration,
unrestricted vocabulary/model confidence, and aggregate request budgets.
`--enable-ocr` enables lines too; `--enable-lines` works without a detector.
The API contract is in [api.md](api.md#cropped-line-recognition).

The native `RecognitionWorker` is shared between direct lines and full-page
ROI recognition. Shared ownership crosses the opaque C ABI safely; destroying
the original recognizer handle does not invalidate a full-page worker. Existing
standalone full-page C entry points remain available. The new Rust server and
native library must ship together.

## Verification

- Rust unit tests: **10 passed**, including line geometry, overflow, RGB/padding
  normalization, and rejection of glyph-only request options.
- Native normalized-tensor parity: **14 cases**, all seven deployed widths
  (128 through 3200), batches 1 and 12. Host-input and GPU-ROI paths returned
  identical text, confidence, timestep class IDs and timestep scores. Shared
  worker lifetime/reclamation checks passed.
- Existing glyph/full-page regression suite: **126 cases matched exactly**
  against the shared-workspace baseline, plus 50 mixed requests. The existing
  short-side detector shape error remains unchanged.
- Direct lines: **100 retained crops**, reordered and compared between batch 1
  and the default grouping. Text/IDs matched; maximum observed score delta was
  zero. This is regression coverage, not independently labelled accuracy.
- **14 width boundaries**, request/chunk boundaries including 128 images and
  partial chunks, and RGB/grayscale/RGBA PNG, JPEG, and PPM inputs passed.
  Generated multi-character `HELLO OCR` text was recognized across formats.
- **10 invalid request cases** passed, including conflicting/empty fields,
  malformed input, unsupported normalized width, invalid batch sizes, unknown
  options, excess image count, and aggregate pixel overflow.
- Separate low-limit tests verified aggregate byte/pixel budgets, image count,
  invalid input late in a request, and successful recovery after errors.
- Full mode: **30 concurrent line/glyph/page requests** produced identical
  results. Detector-free mode: **20 concurrent line/glyph requests** did too.
- Detector-free startup succeeded with `/nonexistent/detector.trt` configured.
  OCR info returned 503. Glyph-only startup returned 503 for both line routes.

## Memory and artifacts

Whole-device GPU snapshots (MiB):

| State | GPU memory |
| --- | ---: |
| Full mode, ready | 1,136 |
| Full mode, lines and ordinary page exercised | 1,162 |
| Full mode, maximum-page storage already resident, lines exercised | 1,204 |
| Detector-free mode, ready | 1,050 |
| Detector-free mode, lines exercised | 1,072 |

Direct line input adds approximately **22 MiB** of host-input GPU staging, not
a third recognition context or classifier buffer set. Memory stayed stable
through repeated mixed requests. The earlier full-page-only high-water mark
was 1,182 MiB; with line staging it becomes 1,204 MiB under the tested bounds.
No simultaneous vLLM load test was performed.

Reports/logs are under `tmp/line-validation/`: `native-parity.log`,
`rust-final.log`, `existing-endpoints.json`, `lines-full-final.json`,
`lines-only-final.json`, `limits.log`, and `disabled.log`. The detector-free
ready measurement is in the initial `lines-only.json`; the final run began
after an earlier test had warmed its input staging. Reports retain aggregates
and response hashes, not source images or recognized text.

Reproduce the line HTTP checks with an isolated server on port 18184:

```bash
uv run --no-project --python /usr/bin/python3 python tools/verify_lines_http.py \
  --crops /home/ekojs/Dev/ocr/tmp/rec-crops \
  --out tmp/line-validation/recheck.json
```

Use `--lines-only` for detector-free checks. The native test target is
`ppocrv6_line_recognizer_smoke MODEL_DIR`. All tests used existing packages and
engine files; no model rebuild or profile tuning was performed.

Local candidate image: `localhost/ppocrv6-vllm-ocr:line-recognition-v1`.
Image ID: `3f3661f48386a11c71cf5155489b9d9fa94dea31209aa3e2225cff10379359d6`.
Native-library SHA-256:
`b62abea56e0418044d1a22dfa25db49823d967583ab68842c1d7dae5a803ddde`.
Server SHA-256:
`0a8c8f4c0f017c00d0c241db277f3624f5c1b73b3b437661396d7c1de8c146ff`.

Development checks caught a duplicated startup block during editing, unavailable
container rustfmt (host rustfmt was used), and an initial test-target lookup
before CMake regeneration. The HTTP harness now waits for readiness after an
early connection-reset race. Its first 128-image case correctly exceeded the
aggregate pixel budget; it was replaced by compact crops to test the independent
image-count boundary. Final checks passed.

The production orchestrator and existing `:dev` image were not changed. No
image was published. Temporary validation containers were stopped/removed.
