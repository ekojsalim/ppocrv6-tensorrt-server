# Model Artifacts

Model files and generated engines are intentionally not committed. Keep them
under `artifacts/ppocrv6-medium/` locally.

Common local layout:

```text
artifacts/
  ppocrv6-medium/
    source/
      metadata.json
      det/
        README.md
        inference.json
        inference.onnx
        inference.yml
      rec/
        README.md
        inference.json
        inference.onnx
        inference.yml
    derived/
      rec-hidden.onnx
      classifier.npz
    classifier/
      weight.fp16.bin
      bias.fp16.bin
      characters.txt
      metadata.json
    engines/
      det-fp16-b1-h256-1280-w256-1280.trt
      rec-hidden-multiprofile-glyph-line.trt
```

Prepare source and derived artifacts in one step:

```bash
python3 -m pip install -e .
python3 tools/prepare_model_artifacts.py
```

That command:

1. downloads the public PP-OCRv6 medium detector/recognizer ONNX files from
   Hugging Face,
2. records source repo metadata under `source/metadata.json`,
3. cuts the recognizer before the classifier into `derived/rec-hidden.onnx`,
4. extracts the classifier into `derived/classifier.npz`,
5. exports native raw classifier files under `classifier/`.

The steps can also be run individually:

```bash
python3 tools/fetch_public_models.py \
  --out-dir artifacts/ppocrv6-medium/source

python3 tools/extract_ppocrv6_classifier.py \
  --model artifacts/ppocrv6-medium/source/rec/inference.onnx \
  --output artifacts/ppocrv6-medium/derived/classifier.npz \
  --hidden-output artifacts/ppocrv6-medium/derived/rec-hidden.onnx

python3 tools/export_classifier_raw.py \
  --classifier artifacts/ppocrv6-medium/derived/classifier.npz \
  --config artifacts/ppocrv6-medium/source/rec/inference.yml \
  --out-dir artifacts/ppocrv6-medium/classifier
```

TensorRT engine build commands depend on the exact ONNX variants, profile
ranges, precision, workspace budget, and target GPU. Prefer rebuilding engines
on the deployment GPU class and writing them under
`artifacts/ppocrv6-medium/engines/`.

## Should We Commit Weights?

The upstream PP-OCRv6 Hugging Face ONNX repositories report `apache-2.0`
license metadata, and PaddleOCR itself is Apache-2.0. Even so, this repository
should not commit model binaries by default:

- normal git handles large binary churn poorly,
- TensorRT engines are GPU/TensorRT-version-specific deployment cache files,
- generated hidden ONNX/classifier files are reproducible from public sources,
- external release assets or Git LFS are easier to replace without source-tree
  noise.

If publishing prebuilt artifacts is useful, prefer a separate release asset or
Git LFS package containing source metadata, checksums, upstream license/model
card copies, and exact generation commands.

## Other Generated Files

The repository also ignores generated smoke samples, native/Rust build outputs,
and local benchmark scratch data:

```text
examples/samples/
native/build*/
server/target/
tmp/
data/
```

Regenerate sample fixtures with:

```bash
python3 examples/generate_samples.py
```
