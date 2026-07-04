# Artifacts

This directory is intentionally ignored except for this README.

Expected local artifacts include:

- PP-OCRv6 detector ONNX.
- PP-OCRv6 recognizer ONNX.
- hidden-recognizer ONNX variant.
- TensorRT detector and recognizer engines.
- raw classifier `weight.fp16.bin`, `bias.fp16.bin`, and `characters.txt`.

See [../docs/model-artifacts.md](../docs/model-artifacts.md) for the expected
layout and helper commands. The default local root is:

```text
artifacts/ppocrv6-medium/
```
