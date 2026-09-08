#!/usr/bin/env python3
"""Exercise the line API with local crops. Persist aggregates/hashes, not text."""
import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
import time
from pathlib import Path
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont
from verify_workspace_http import encode, gpu_mib, semantic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18184")
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lines-only", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + 20
    while True:
        try:
            with urllib.request.urlopen(args.base_url + "/health", timeout=1) as response:
                assert response.status == 200
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)
    report = {"ready_gpu_mib": gpu_mib()}

    def call(payload=None, path="/v1/lines/recognize", expected=200):
        request = urllib.request.Request(args.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        assert status == expected, (path, status, expected, raw[:200] if status != 200 else "")
        return json.loads(raw) if status == 200 else None

    info = call(path="/v1/lines/info")
    report["buckets"] = info["recognition_buckets"]
    # There are only two recognition workers (glyph and shared line/page).
    assert info["recognizer"]["shared_engine_use_count"] == 2
    if args.lines_only:
        call(path="/v1/ocr/info", expected=503)
    images = [Image.open(p).convert("RGB") for p in sorted(args.crops.glob("*.png"))]
    assert len(images) == 100
    accepted = [image for image in images if (image.width * 48 + image.height - 1) // image.height <= 3200]
    encoded = [encode(image) for image in accepted]
    report["crop_count"] = len(encoded)
    result = call({"images": encoded})
    assert result["count"] == len(encoded)
    assert [p["index"] for p in result["predictions"]] == list(range(len(encoded)))
    report["crop_result_sha256"] = hashlib.sha256(json.dumps(semantic(result), sort_keys=True).encode()).hexdigest()
    # Change grouping and order, checking text/IDs separately from scores, which
    # can differ slightly when TensorRT changes batch shape.
    reversed_result = call({"images": list(reversed(encoded)), "batch_size": 1})
    score_delta = 0.0
    for a, b in zip(result["predictions"], reversed(reversed_result["predictions"])):
        assert a["text"] == b["text"] and a["class_ids"] == b["class_ids"], "batch/order text mismatch"
        score_delta = max(score_delta, abs(a["score"] - b["score"]))
    report["max_batch_score_delta"] = score_delta
    compact = encode(Image.new("RGB", (128, 48), "white"))
    maximum = call({"images": [compact] * 128})
    assert maximum["count"] == 128
    assert len(maximum["recognition_chunks"]) == 11
    assert all(c["batch"] <= 12 for c in maximum["recognition_chunks"])
    for batch in (2, 11):
        response = call({"images": [compact] * 13, "batch_size": batch})
        assert response["count"] == 13
        assert all(p["text"] == maximum["predictions"][0]["text"] for p in response["predictions"])
    report["request_batch_boundaries"] = [1, 2, 11, 12, 13, 128]
    boundaries = [1, 128, 129, 640, 641, 960, 961, 1280, 1281, 1600, 1601, 2400, 2401, 3200]
    for width in boundaries:
        image = encode(Image.new("RGB", (width, 48), "white"))
        result = call({"image": image})
        assert result["prediction"]["bucket_width"] == next(b for b in report["buckets"] if b >= width)
        assert result["prediction"] == result["predictions"][0]
    report["boundary_cases"] = len(boundaries)
    font = ImageFont.truetype("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf", 32)
    text_image = Image.new("RGB", (256, 48), "white")
    ImageDraw.Draw(text_image).text((0, 0), "HELLO OCR", font=font, fill="black")
    formats = []
    for mode, fmt in [("RGB", "PNG"), ("L", "PNG"), ("RGBA", "PNG"), ("RGB", "JPEG"), ("RGB", "PPM")]:
        buf = io.BytesIO()
        text_image.convert(mode).save(buf, format=fmt)
        value = "data:image/" + fmt.lower() + ";base64," + base64.b64encode(buf.getvalue()).decode()
        response = call({"image": value})
        assert response["prediction"]["text"].replace(" ", "") == "HELLOOCR"
        formats.append(mode + "/" + fmt)
    report["formats"] = formats
    for payload, status in [({}, 400), ({"images": []}, 400),
            ({"image": encoded[0], "images": encoded[:1]}, 400),
            ({"images": [encoded[0]] * 129}, 413), ({"image": "??"}, 400),
            ({"image": encoded[0], "width": 80}, 422),
            ({"image": encoded[0], "batch_size": 13}, 400),
            ({"image": encoded[0], "batch_size": 0}, 400),
            ({"image": encode(Image.new("RGB", (3201, 48), "white"))}, 400),
            ({"images": [encode(Image.new("RGB", (4000, 2500), "white"))] * 2}, 413)]:
        call(payload, expected=status)
    report["invalid_cases"] = 10
    # Same inputs/chunk shapes must remain exactly stable while sharing the
    # recognition context with full-page requests and the workspace with glyphs.
    line_payload = {"images": encoded[:20]}
    glyph_payload = {"image": encode(text_image), "width": 80, "character_policy": "all"}
    tasks = [("/v1/lines/recognize", line_payload), ("/v1/glyphs/recognize", glyph_payload)]
    if not args.lines_only:
        page = Image.new("RGB", (1280, 960), "white")
        draw = ImageDraw.Draw(page)
        for y in range(20, 900, 60):
            draw.text((20, y), "Concurrent OCR validation 0123456789", font=font, fill="black")
        tasks.append(("/v1/ocr/recognize", {"image": encode(page)}))
    expected = {path: semantic(call(payload, path)) for path, payload in tasks}
    report["warm_gpu_mib"] = gpu_mib()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [(path, executor.submit(call, payload, path)) for _ in range(10) for path, payload in tasks]
        for path, future in futures:
            assert semantic(future.result()) == expected[path], "concurrent result mismatch"
    report["concurrent_requests"] = len(futures)
    report["final_gpu_mib"] = gpu_mib()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
