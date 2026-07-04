# Examples

Generate dependency-free sample images:

```bash
python3 examples/generate_samples.py
```

This writes:

```text
examples/samples/glyph_A.ppm
examples/samples/page_ocr_portrait.ppm
examples/samples/page_ocr_landscape.ppm
```

Post a generated glyph sample:

```bash
python3 examples/post_image.py \
  --kind glyph \
  --image examples/samples/glyph_A.ppm
```

Post a generated page sample:

```bash
python3 examples/post_image.py \
  --kind ocr \
  --image examples/samples/page_ocr_portrait.ppm
```

Run the assertive end-to-end smoke check against a running server:

```bash
python3 examples/e2e_smoke.py \
  --base-url http://127.0.0.1:8184
```

The smoke check asserts that the health endpoint responds, the glyph fixture
decodes as `A`, and both portrait and landscape page fixtures find stable
tokens such as `HELLO OCR`, `TEST`, and `123`.

The generated images are smoke-test fixtures, not accuracy benchmarks. They
exercise the HTTP/base64/image-decode/native-runtime path without requiring
external sample data.
