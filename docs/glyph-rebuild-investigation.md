# Glyph differences after the recognition-engine rebuild

Investigated 2026-09-06. The four text changes are reproducible, close-ranking
flips originating in the TensorRT hidden output. All four fixture labels are
absent from the classifier vocabulary, so neither engine can emit the intended
character. This is not evidence of four newly incorrect predictions.

## What changed

The same `line-recognition-v1` server image was used with the original engine
and the experimental three-profile engine. Each request contained the same
1,024 sorted RGB glyph fixtures, batch size 256, width 80, `character_policy=all`,
`score_mode=model`, and timestep output enabled. Three identical requests per
engine produced exactly identical predictions, including all timestep scores.
Repeating the native dumps in separate processes also reproduced all eight
hidden-tensor SHA-256 hashes exactly (four batches per engine).

| Fixture label | Old winner / probability | Old runner-up / probability | New winner / probability | New runner-up / probability |
| --- | --- | --- | --- | --- |
| U+4E85 (亅) | 」 / 0.485378 | J / 0.477544 | J / 0.486885 | 」 / 0.476304 |
| U+4EF8 (仸) | 沃 / 0.302717 | 任 / 0.291214 | 任 / 0.294764 | 沃 / 0.292240 |
| U+4F2E (伮) | 俶 / 0.240180 | 做 / 0.230946 | 做 / 0.240298 | 俶 / 0.236118 |
| U+4FF9 (俹) | 亞 / 0.182772 | 婭 / 0.177783 | 婭 / 0.178489 | 亞 / 0.177009 |

Probabilities above were independently calculated in float64 from the saved
FP16 hidden outputs and deployed FP16 classifier weights/biases. The original
winner margins are only 0.50–1.15 percentage points (0.0163–0.0392 logits).
Every change is at timestep 3 (zero-based), with the remaining timesteps blank.
The first fixture was visually inspected: it contains an actual hook-shaped
glyph, consistent with its confusion with punctuation/J, rather than a tofu box.
The other fixture labels have not been independently visually adjudicated.

## Locating the difference

Added optional `--out-hidden` to the existing native classifier diagnostic.
It saves raw FP16 hidden values after the completed execution, without modifying
inference behavior when the flag is absent.

All fixtures are 48×48 RGB. Native diagnostic input uses the server's exact
no-resize normalization `(float32(pixel)/255 - 0.5)/0.5`, with zero padding to
width 80, retaining all four original 256-image batches and positions. Native
WMMA classifier IDs match HTTP IDs at **all 10,240 timesteps for each engine**.

An independent float64 matrix multiply plus bias and softmax over all 18,710
classes also matches every native timestep winner for both engines. Its maximum
winning-probability difference from the native classifier is 0.00000121.
Consequently, the observed flips already exist in the backbone hidden tensors;
neither the custom classifier nor CTC decoding introduces them in this check.
This reference increases classifier arithmetic precision only: it is not an
FP32/FP64 reference for the backbone, whose outputs and weights remain FP16.

Across 1,966,080 hidden values per engine:

- Both saved outputs contain only finite values.
- Difference RMS: 0.0148924, versus original hidden RMS 2.96393 (about 0.50%).
- Mean absolute difference: 0.0105839; maximum: 0.28125.

The prior old-line-bucket control reproduced all four changes. Batch 256 can
only use the glyph profile, so the new short-line bucket selection cannot
explain these glyph changes.

## What the accuracy result means

Only **751 of the 1,024 filename labels exist in the deployed vocabulary**;
273 are unrepresentable. All four changed labels are among those 273.
Both engines score 559/751 (74.43%) on supported filename labels and 559/1024
(54.59%) across the complete set. Supported-label predictions did not change.
These are synthetic-font, filename-label results, not production accuracy.
The earlier 559/1024 total mixed recognition quality and vocabulary coverage;
future quality reports should report those separately.

## Interpretation and next engine work

TensorRT documents that independent builds need not be bit-identical: tactic
selection and floating-point accumulation can change. See NVIDIA's
[precision and reproducibility guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html).
The evidence here is consistent with those numerical changes crossing close
classification boundaries. It does not identify the exact differing tactic or
layer, or prove that all possible inputs retain accuracy. Finite final tensors
also do not prove that every intermediate operation was finite.

The experimental build warned about FP16 layer normalization. That warrants
an accuracy experiment but is not proof that layer normalization caused these
four differences. Before selecting a rebuilt engine, compare against a
higher-precision backbone on representative labeled lines/pages and supported
glyphs; inspect sensitive reductions/normalization if differences warrant it.
Record build configuration, environment, ONNX and classifier hashes, and a
reproducible tactic/timing cache alongside selected engine artifacts.

For strict glyph compatibility during line-profile experiments, keep glyph
requests on the original serialized engine. A separate experimental line
engine may cost extra memory and needs measurement. The selected CPU
preprocessing optimization already retains the original engine and had exact
text/confidence parity in its regression set. No deployment change is needed
for this investigation.

## Local reproduction artifacts

`tmp/glyph-rebuild-investigation-20260906/` contains:

- `probe.py`, `old.json`, `new.json`: repeated HTTP outputs for synthetic fixtures.
- `inputs.py`, `input{0..3}.bin`: exact padded float32 inputs.
- `dump.sh`, `{old,new}{0..3}.{hidden,ids,prob,log}`: native runs and tensors.
- `analyze.py`, `analysis.json`: hidden deltas and top-five probabilities.
- `check_all.py`, `all-timesteps.json`: independent classifier checks.

The host analysis uses temporary `uv run --no-project --with numpy --with pillow`
environments; project dependencies were not changed. The native tool was built
using cached builder image `ec7b39780087` with benchmark tools enabled. All
isolated HTTP servers were stopped. The deployed engine/configuration and
selected image remain unchanged.
