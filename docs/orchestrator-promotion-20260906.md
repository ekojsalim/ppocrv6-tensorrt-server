# Orchestrator promotion and OCR/vLLM co-residency

This records the initial promotion on September 6. The subsequent
[detector-padding fix](detector-padding-validation.md) supersedes the image
selection below and resolves the short-side error mentioned in these results.
Image tags and filesystem paths in this report refer to the local validation
environment; the compiled artifacts and raw fixtures are not part of this repository.

Promoted locally on 2026-09-06 after acceptance of the rebuilt recognition
engine's investigated numerical differences. `serve-orchestrator/serve.toml`
continues to use `localhost/ppocrv6-vllm-ocr:dev` and the existing mounted model
directory. No registry publication was performed.

## Selected artifacts

- OCR `:dev` now references image
  `701f463e2a48298f78e9b64ffdbd98bf0b7c0c65721bf7d5147a98dd1ec2285a`
  (also tagged `:line-preprocess-v2`).
- Server SHA-256:
  `a04f1e8c4257a936dd20c974bd638f3dcb3cea88311367a28e778fe0ec5006f1`.
- Native library SHA-256:
  `b62abea56e0418044d1a22dfa25db49823d967583ab68842c1d7dae5a803ddde`.
- Mounted recognition engine SHA-256:
  `7977ff9bcca78eb28cf9f972ff0cc67b04f7cc5edbf8b034891ceb76b5052125`.
  This replaces `tmp/vllm-ocr-model/engines/rec-hidden-multiprofile-glyph-line.trt`.
- The image supplies shared enqueue workspace, the shared cropped-line API,
  and parallel CPU line preprocessing. The separately mounted engine supplies
  the glyph/short-line/long-line profiles. Enabled line buckets are
  128, 256, 384, 512, 640, 960, 1280, 1600, 2400, 3200.

The vLLM configuration is unchanged: 9,506,652,160 KV bytes, 32 maximum
sequences, the existing pinned v0.26 runtime and optimized production bundle,
with k=4 MTP. All requests used synthetic token IDs or retained OCR fixtures;
generated language-model text was discarded.

## Validation

The standalone promoted OCR suite passed 126 page/glyph cases (124 successful
requests and the two established invalid-input/short-side error controls),
50 concurrent mixed requests, and the line suite with 100 retained crops,
14 width boundaries, 10 invalid cases, and 30 concurrent requests.
It covers 16-million-pixel page allocation, 1,024-glyph requests at width 128,
and width-3200 line batches of 12.

Whole-device OCR-only memory was 1,138 MiB ready, 1,186 MiB after the page/glyph
suite, and 1,208 MiB after the line suite. The older full-OCR build measured
1,728 MiB after page/glyph warmup. The selected build therefore saves about
520 MiB even after the additional line endpoint staging allocation.

vLLM then started with fully warmed OCR resident. Repeating the page/glyph
suite produced identical semantic hashes to the OCR-only control; the line
suite passed as well. OCR was subsequently stopped and started with vLLM
resident, and mixed traffic immediately exercised page, glyph, and line paths.
This validates both startup orders and lazy source/staging allocations.

Three overlapping-load runs each completed 32 vLLM requests and 120 OCR
requests. OCR rotated historical/dense/long pages, the 16-million-pixel blank
page, 1,024 glyphs at width 128, and 12 cropped lines at width 3200.

| vLLM workload (input/output tokens per request) | Maximum active sequences | Maximum KV usage | Lowest sampled free VRAM |
| --- | ---: | ---: | ---: |
| Mixed 128/512/1024/2048 inputs, 512 outputs | 21 | 90.50% | 236 MiB |
| 128 inputs, 1024 outputs; OCR restarted while vLLM resident | 20 | 87.00% | 236 MiB |
| 128 inputs, 256 outputs | 32 | 44.82% | 236 MiB |

All 96 language-model requests completed their requested output lengths
(57,344 generated tokens total). All 360 overlapping OCR requests succeeded
with stable semantic hashes per case. There were no request errors or vLLM
preemptions, and no OOM occurred. Admission limited simultaneous execution of
the longer requests; the final shorter workload actually reached 32 active
sequences. The two limits (high KV utilization and 32 active sequences) were
tested separately, not achieved simultaneously in one workload.

See `stress.json`, `stress-short.json`, and `stress-tiny.json` for results.
Memory was sampled with `nvidia-smi` at approximately 0.2-second intervals;
this is a sampled minimum, not a CUDA allocation trace or a guarantee for all
production traffic. All tests preserve the existing admission policy.

The previously identified lazy full-OCR workspace OOM did not reproduce in
these checks. The shared workspace reserves the maximum requirement at OCR
startup, and full OCR no longer adds a separate long-line workspace. Remaining
free VRAM is still modest; the tests establish this configuration's observed
behavior rather than an unlimited memory safety guarantee. The known detector
minimum-short-side error remains separate from memory exhaustion.

## Cleanup and rollback

Superseded OCR image tags and unused OCR build images were removed. Podman
reported image storage decreasing from 151.8 GB to 148.7 GB (about 3.1 GB).
Shared layers mean displayed per-image sizes must not be added together.
Unrelated service images and volumes were preserved. The current TensorRT
base/builder cache was retained; one older image referenced by an external
build container was also retained rather than force-removing that container.

Rollback requires both the image and mounted engine:

```bash
cd /home/ekojs/Dev/serve-orchestrator
uv run serve-node down-ocr
podman tag localhost/ppocrv6-vllm-ocr:rollback-pre-20260906 localhost/ppocrv6-vllm-ocr:dev
cp /home/ekojs/Dev/ppocrv6-tensorrt-server/tmp/promotion-20260906/rollback/rec-hidden-multiprofile-glyph-line.trt /home/ekojs/Dev/ppocrv6-tensorrt-server/tmp/vllm-ocr-model/engines/rec-hidden-multiprofile-glyph-line.trt
uv run serve-node up-ocr --no-tmux
```

The old image is `93b0cb235248cfde013c2b3db16cfcbe6539e344fe2d740248287b959ccb37ac`;
the old engine SHA-256 is
`fcedf0387857832e9e7a31b3102f88df9ebe242c53d1d7fc06d827136abe67be`.
Rollback restores the older memory behavior as well as the older outputs.

## Artifacts

`tmp/promotion-20260906/` holds the rollback engine/configuration, image
inventories, cleanup log, Podman storage snapshots, startup logs, line/profile
metadata, standalone and co-resident verification JSON/logs, and `stress.py`
with aggregate concurrent-load results. The stress script uses standard HTTP
APIs and system Pillow via `uv run --no-project --python /usr/bin/python3 python`.
No project dependencies were changed.
