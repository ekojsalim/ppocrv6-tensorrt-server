# Line profile and preprocessing optimization

Measured 2026-09-06 Asia/Jakarta on the local RTX 5090. During these experiments,
production settings, model files, and the existing line-recognition-v1 image
were unchanged. The subsequent promotion is recorded below.

## Selected change: parallel line resize

Large multi-image line chunks now resize/normalize independent images in
parallel using the existing Rayon pool. Small/single-image chunks stay serial.
The normalized tensor still holds only one inference chunk, and each task
writes a disjoint image slice. No engine or interpolation arithmetic changed.
Timing metadata now separates decode, resize, native-call, and native-predict
time so future tuning can target the remaining work.

Median localhost HTTP latency, three warmups and 15 measured requests per case,
one server permit, original two-profile engine in both runs:

| Workload | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| 100 retained crops, chunk 4 | 128.27 ms | 84.27 ms | 34.3% |
| 100 retained crops, chunk 8 | 123.28 ms | 74.72 ms | 39.4% |
| 100 retained crops, chunk 12 | 121.86 ms | 70.30 ms | 42.3% |
| Synthetic width 3000, batch 12 | 13.69 ms | 12.32 ms | 10.0% |
| Historical full page | 45.33 ms | 45.48 ms | unchanged within noise |
| 1,024 glyphs | 44.95 ms | 45.35 ms | unchanged within noise |

All benchmark text hashes and confidence scores matched exactly. GPU memory was
identical: 1,136 MiB ready and 1,164 MiB after this benchmark mix (whole device).
A separate eight-request phase sample for the 100-crop request measured about
10.3 ms decode, 17.9 ms resize, 38.0 ms native call (36.4 ms native predict), and
66.6 ms total handler time. Native predict is a subset of native-call time.
These phase medians are from a separate sample and are not the HTTP table's
end-to-end measurements.

Validation:

- 10 Rust tests passed.
- Line regression covered 100 crops, all width buckets, request/batch limits,
  formats and invalid input, plus 30 mixed line/glyph/page requests with four
  server permits. Text/score parity remained exact and memory stayed stable.
- Existing glyph/full-page regression: 126 cases and 50 concurrent requests,
  compared against the prior line-endpoint build.

## Experimental engine: short-line profile

Built a separate FP16 recognition engine from the existing ONNX with the existing
builder at optimization level 3 and a 1,024-MiB builder workspace limit:

| Profile | Min (batch,width) | Opt | Max |
| --- | --- | --- | --- |
| Glyph | 1,48 | 128,80 | 256,128 |
| Short line | 1,128 | 8,384 | 12,640 |
| Long line | 1,640 | 8,1600 | 12,3200 |

The new profile enables buckets 256, 384, and 512. Compared with the old engine
and server, synthetic short-line batch-12 latency fell 43.2% at source width
160, 24.5% at 320, and 15.6% at 480. Long-line changes were small. Mixed 100-crop
latency was essentially unchanged (+1.2% at chunk 12); the historical page
showed no established gain.

A control using the rebuilt engine but retaining the old bucket list reduced
those short-line batch-12 times only 2.8–4.5%, supporting reduced padding as the
main benefit rather than a general engine speedup.

The rebuilt engine returned unchanged line/page text in these samples, but
changed four of 1,024 synthetic glyph outputs. All four were incorrect under
both engines against filename-derived codepoint labels; exact-match totals
stayed 559/1,024. This is a narrow synthetic-font check, not production accuracy.
Confidence also changed (maximum observed glyph difference about 0.0212).
The old-bucket control reproduced those four glyph differences, so they come
from the rebuild rather than line bucketing.

Follow-up [numerical investigation](glyph-rebuild-investigation.md) located all
four flips in close classifier rankings caused by changed hidden tensors. All
four intended labels are absent from the model vocabulary. Only 751/1,024
fixture labels are representable; both engines score 559/751 on that subset,
with no supported-label text changes. Repeated requests are deterministic.

The short profile requires only 67.5 MiB of enqueue memory, but the long profile
still requires 337.5 MiB. The shared arena therefore does not shrink. This
benchmark's warm GPU use increased from 1,164 to 1,168 MiB.

**Initial decision: the rebuilt engine was retained as an experiment.**
It helps short-line-heavy workloads, but did not improve this mixed workload and
does not preserve glyph outputs exactly. The selected preprocessing change uses
the original engine and has broader measured benefit.

Subsequent decision on 2026-09-06: after the numerical investigation and user
acceptance, the rebuilt engine was promoted alongside the preprocessing
optimization. See [promotion and co-residency validation](orchestrator-promotion-20260906.md)
for the current selection; the measurements above retain their original controls.

## Artifacts and reproduction

Artifacts: `tmp/line-profile-tuning-20260906/`:

- `baseline.json`, `three-profile.json`, `three-profile-old-buckets.json`,
  `parallel-preprocess.json`: benchmarks and per-output hashes/scores.
- `engine-build.log`, `rec-three-profile.trt`: experimental engine only.
- `preprocess-phases.json`, `parallel-lines-regression.json`,
  `parallel-existing-regression.json`: phase measurements and validation.

Benchmark command for each isolated server on localhost:18184:

```bash
uv run --no-project --python /usr/bin/python3 python tools/bench_line_profiles.py \
  --fixtures /home/ekojs/Dev/ocr --out tmp/line-profile-tuning-20260906/recheck.json \
  --compare tmp/line-profile-tuning-20260906/baseline.json
```

Selected local image: `localhost/ppocrv6-vllm-ocr:line-preprocess-v2`.
Image ID: `701f463e2a48298f78e9b64ffdbd98bf0b7c0c65721bf7d5147a98dd1ec2285a`.
Server SHA-256: `a04f1e8c4257a936dd20c974bd638f3dcb3cea88311367a28e778fe0ec5006f1`.
Original engine SHA-256 remains
`fcedf0387857832e9e7a31b3102f88df9ebe242c53d1d7fc06d827136abe67be`.
ONNX SHA-256: `208f1bc261966dd9eefc85b5599b4cf2ae04330504fb93d7cb5e7faf76d54652`.

An early benchmark attempt raced server readiness; the harness now waits for
health before starting. No packages were installed or upgraded. No production
service was changed, no image was published, and no vLLM co-residency test was
performed. Next engine/kernel tuning should separate TensorRT and classifier
time within native recognition before changing long-line tactics.
