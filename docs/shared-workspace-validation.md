# Shared TensorRT workspace validation

Measured on 2026-09-06 Asia/Jakarta, RTX 5090, existing TensorRT 10.14/CUDA 13
OCR engines. No engine rebuild or model/recognition-policy change.

## Result

| GPU memory (whole device) | Baseline | Shared workspace |
| --- | ---: | ---: |
| Ready | 1,344 MiB | 1,136 MiB |
| After full fixture suite | 1,728 MiB | 1,182 MiB |
| After repeated mixed traffic | 1,728 MiB | 1,182 MiB |

Warmed saving: **546 MiB (31.6%)**. Detector and both recognition contexts
share a 353,909,760-byte arena instead of retaining independent enqueue memory.
GPU-input full-page paths also avoid allocating unused host-input staging
buffers. The arena is reserved at startup; page source storage can still grow
by approximately 46 MiB at the configured pixel limit.

## Fixtures and parity

The older local OCR checkout retains 1,024 synthetic CJK glyphs in each of RGB
and grayscale form, 19 font glyphs, 100 recognition crops, and a full-page
benchmark sample. The current checkout retains generated full-page smoke PPMs.
These provide useful regression coverage, not an independently labelled
production accuracy benchmark.

`tools/verify_workspace_http.py` exercised 126 distinct request cases:

- All 2,048 synthetic glyph images at widths 48, 80, 128, in requests of 1,024
  images using inference chunks of 256.
- Font glyphs under all four character policies and both score modes.
- All 100 old recognition crops padded to valid detector dimensions, the
  historical page, generated smoke pages, and dense/long/portrait synthetic pages.
- Every deployed recognition bucket: 128, 640, 960, 1280, 1600, 2400, 3200.
- Blank pages through 4000x4000 (16 million pixels), invalid image data, and
  the known unsupported 192x640 detector shape.

124 cases succeeded; invalid data returned 400 and the known short-side shape
returned 500 in both builds. The shape bug is separate from this change.
All semantic JSON hashes matched exactly, including boxes, text, class IDs,
scores, and chunk metadata. Only timings and throughput fields were excluded.
Reports contain hashes and counts, not recognized text or image payloads.

Repeated mixed traffic added 50 requests from four client threads. Tests ran
with both four server permits (native concurrency safety) and the production
default of one. No mismatches or post-warmup memory growth occurred.

Native tests passed in shared and independent modes: concurrent workspace
writes, growth, ownership/lifetime, and host-exception stream draining. Existing
CTC and glyph-shape smoke tests also passed. Build used OpenCV/Clipper, CUDA
architecture 120, and the cached TensorRT 10.14 builder image.

## Performance

Median localhost end-to-end latency, 12 warm repetitions, one server permit:

| Workload | Baseline | Shared |
| --- | ---: | ---: |
| 1,024 CJK glyphs, width 80 | 43.18 ms | 42.66 ms |
| 19 font glyphs | 2.74 ms | 2.82 ms |
| Historical full page | 44.46 ms | 44.63 ms |
| Dense synthetic page | 11.51 ms | 11.55 ms |
| Long-line synthetic page | 9.21 ms | 9.11 ms |

The 50-request mixed workload took 1,070.48 vs 1,073.36 ms with one permit
(+0.27%). With four permits it took 773.17 vs 813.29 ms (+5.2%). Shared workspace
serializes GPU stages; use independent arenas when additional parallel GPU
throughput is more valuable than memory. These are bounded local measurements,
not a production latency or sustained-throughput guarantee.

## Reproduction and artifacts

Reports and build logs: `tmp/workspace-validation/`:
`baseline-final.json`, `candidate-final.json` (four permits),
`baseline-permit1.json`, `candidate-permit1.json`, and `native-final.log`.

For each isolated server, run with system Pillow already installed:

```bash
uv run --no-project --python /usr/bin/python3 python tools/verify_workspace_http.py \
  --fixtures /home/ekojs/Dev/ocr --out tmp/workspace-validation/new.json \
  --compare tmp/workspace-validation/baseline-permit1.json
```

Local candidate image: `localhost/ppocrv6-vllm-ocr:shared-workspace-v1`.
Image ID: `20518834eb2be62b32ac4b1f4f142e2a24f1254ca602176935f97cd67b874624`.
Baseline image ID: `93b0cb235248cfde013c2b3db16cfcbe6539e344fe2d740248287b959ccb37ac`.
Candidate native-library SHA-256:
`0d6351e64d3334f769e0766d13ec53df0829fdc154f5224225ba621ea184fc78`.
The candidate layers the rebuilt native library onto the unchanged baseline
image; Rust server, engine files, and runtime libraries are unchanged.

Initial harness hashes accidentally included derived throughput fields, causing
false mismatches even within the baseline. Corrected before the final A/B runs.
The initial native-test invocation omitted CMake's `bin/` output directory;
the corrected commands passed. No runtime implementation test failed.

No production service/configuration was changed, no image was published, and
temporary validation servers were removed. Co-resident vLLM validation is still
needed before declaring the existing KV budget safe; these runs isolated OCR.
