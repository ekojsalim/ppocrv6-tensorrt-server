#!/usr/bin/env python3
"""Measure engine/profile candidates without persisting recognized source text."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont
from verify_workspace_http import encode, gpu_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18184")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    deadline = time.monotonic() + 20
    while True:
        try:
            with urllib.request.urlopen(args.base_url + "/health", timeout=1):
                break
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)

    def request(path, payload=None):
        request = urllib.request.Request(args.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    report = {"ready_gpu_mib": gpu_mib(), "info": request("/v1/lines/info"), "cases": {}}
    cases = []
    font = ImageFont.truetype("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf", 32)
    for width in [160, 320, 480, 800, 1200, 2000, 3000]:
        label = ""
        for c in "HELLO OCR 0123456789 " * 15:
            if font.getlength(label + c) > width - 12:
                break
            label += c
        label = label.rstrip()
        image = Image.new("RGB", (width, 48), "white")
        ImageDraw.Draw(image).text((4, 0), label, font=font, fill="black")
        for batch in [1, 8, 12]:
            cases.append((f"synthetic-w{width}-b{batch}", "/v1/lines/recognize",
                {"images": [encode(image)] * batch, "batch_size": batch}, label))
    crops = [encode(Image.open(p)) for p in sorted((args.fixtures / "tmp/rec-crops").glob("*.png"))]
    assert len(crops) == 100
    for batch in [4, 8, 12]:
        cases.append((f"crops100-b{batch}", "/v1/lines/recognize", {"images": crops, "batch_size": batch}, None))
    cases.append(("historical-page", "/v1/ocr/recognize", {"image": encode(Image.open(args.fixtures / "tmp/ppocrv6-bench-sample.png"))}, None))
    glyphs = [encode(Image.open(p)) for p in sorted((args.fixtures / "tmp/synthetic-cjk-glyphs/rgb").glob("*.png"))]
    cases.append(("glyph1024", "/v1/glyphs/recognize", {"images": glyphs, "batch_size": 256,
        "width": 80, "character_policy": "all", "score_mode": "model"}, None))
    for name, path, payload, label in cases:
        for _ in range(3):
            request(path, payload)
        durations, server_durations = [], []
        for _ in range(args.repeats):
            started = time.perf_counter()
            response = request(path, payload)
            durations.append((time.perf_counter() - started) * 1000)
            server_durations.append(response.get("timing", {}).get("total_ms", response.get("elapsed_ms", 0)))
        predictions = response.get("predictions", response.get("lines", []))
        item = {"median_ms": statistics.median(durations), "min_ms": min(durations),
            "server_median_ms": statistics.median(server_durations), "count": len(predictions),
            "text_hashes": [hashlib.sha256(p["text"].encode()).hexdigest() for p in predictions],
            "scores": [p["score"] for p in predictions],
            "buckets": [p.get("bucket_width") for p in predictions]}
        if label is not None:
            item["expected_text_matches"] = sum(p["text"].replace(" ", "") == label.replace(" ", "") for p in predictions)
        report["cases"][name] = item
    report["final_gpu_mib"] = gpu_mib()
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        report["comparison"] = {}
        for name, item in report["cases"].items():
            control = baseline["cases"][name]
            assert item["count"] == control["count"], name
            report["comparison"][name] = {
                "latency_change_pct": 100 * (item["median_ms"] / control["median_ms"] - 1),
                "text_changes": sum(a != b for a, b in zip(item["text_hashes"], control["text_hashes"])),
                "max_score_delta": max((abs(a - b) for a, b in zip(item["scores"], control["scores"])), default=0),
            }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ready_gpu_mib": report["ready_gpu_mib"], "final_gpu_mib": report["final_gpu_mib"],
        "comparison": report.get("comparison", {}),
        "medians_ms": {name: round(item["median_ms"], 3) for name, item in report["cases"].items()}}, indent=2))


if __name__ == "__main__":
    main()
