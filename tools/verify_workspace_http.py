#!/usr/bin/env python3
"""Compare semantic response hashes and timings across isolated OCR builds.

Requires Pillow and the explicitly supplied local fixture tree. Reports contain
hashes and aggregate counts, not recognized text or image payloads.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont


def encode(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode()


def semantic(value):
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items()
                if k not in {"timing", "timings"} and
                not k.endswith(("_ms", "_timings", "_per_second"))}
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def request(base, endpoint, payload=None):
    req = urllib.request.Request(base + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            status, result = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status, result = error.code, json.load(error)
    elapsed = (time.perf_counter() - started) * 1000
    digest = hashlib.sha256(json.dumps(semantic(result), sort_keys=True).encode()).hexdigest()
    return {"status": status, "sha256": digest, "client_ms": elapsed,
            "count": result.get("count"),
            "buckets": sorted({line.get("bucket_width", 0) for line in result.get("lines", [])})}


def gpu_mib():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
               "--format=csv,noheader,nounits"], text=True).strip())


def cases(root):
    glyph = "/v1/glyphs/recognize"
    page = "/v1/ocr/recognize"
    result = []
    def add(name, endpoint, payload):
        result.append((name, endpoint, payload))
    for kind in ("rgb", "gray"):
        paths = sorted((root / "tmp/synthetic-cjk-glyphs" / kind).glob("*.png"))
        assert len(paths) == 1024, (kind, len(paths))
        imgs = [encode(Image.open(p)) for p in paths]
        for width in (48, 80, 128):
            add(f"cjk-{kind}-w{width}", glyph, {"images": imgs, "width": width,
                "batch_size": 256, "character_policy": "all", "score_mode": "model"})
    paths = sorted((root / "data/glyphs").rglob("*.png"))
    imgs = [encode(Image.open(p)) for p in paths]
    assert len(imgs) >= 19
    for policy in ("all", "suppress_ascii", "cjk_focus", "cjk_focus_fallback"):
        for mode in ("model", "accepted"):
            add(f"font-{policy}-{mode}", glyph, {"images": imgs, "width": 80,
                "batch_size": 8, "character_policy": policy, "score_mode": mode})
    for p in sorted((root / "tmp/rec-crops").glob("*.png")):
        # These are line crops, so run them through the full pipeline on a
        # padded canvas that respects the existing detector's 256 minimum.
        im = Image.open(p).convert("RGB")
        canvas = Image.new("RGB", (max(256, im.width), max(256, im.height)), "white")
        canvas.paste(im, (0, 0))
        add(f"crop-{p.stem}", page, {"image": encode(canvas)})
    local = Path(__file__).resolve().parents[1] / "examples/samples"
    for p in sorted(local.glob("page*.ppm")):
        add(p.stem, page, {"image": encode(Image.open(p))})
    p = root / "tmp/ppocrv6-bench-sample.png"
    add("historical-page", page, {"image": encode(Image.open(p))})
    font_path = "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"
    for width, height, size, label in [(1280, 960, 32, "dense"),
            (1280, 256, 14, "long"), (256, 1280, 24, "portrait")]:
        im = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font_path, size)
        for y in range(16, height - size, size * 2):
            draw.text((8, y), "Synthetic OCR 0123456789 " * 12, font=font, fill="black")
        add(label, page, {"image": encode(im)})
    for width, height in [(256, 256), (1280, 1280), (4000, 4000), (192, 640)]:
        add(f"blank-{width}x{height}", page,
            {"image": encode(Image.new("RGB", (width, height), "white"))})
    add("invalid-image", page, {"image": "bm90IGFuIGltYWdl"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18184")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    workload = cases(args.fixtures)
    report = {"ready_gpu_mib": gpu_mib(), "cases": {}}
    for name, endpoint, payload in workload:
        report["cases"][name] = request(args.base_url, endpoint, payload)
    report["warm_gpu_mib"] = gpu_mib()
    # Repeated mixed requests use distinct worker mutexes and TRT profiles.
    selected = [x for x in workload if x[0] in {"dense", "long", "historical-page", "font-all-model", "cjk-rgb-w80"}]
    report["concurrent_mismatches"] = []
    concurrent_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [(name, executor.submit(request, args.base_url, endpoint, payload))
                   for _ in range(10) for name, endpoint, payload in selected]
        for name, future in futures:
            value = future.result()
            if value["sha256"] != report["cases"][name]["sha256"]:
                report["concurrent_mismatches"].append(name)
    report["concurrent_requests"] = len(futures)
    report["concurrent_wall_ms"] = (time.perf_counter() - concurrent_started) * 1000
    report["bench"] = {}
    for name, endpoint, payload in selected:
        values = [request(args.base_url, endpoint, payload)["client_ms"] for _ in range(12)]
        report["bench"][name] = {"median_ms": statistics.median(values), "min_ms": min(values)}
    report["final_gpu_mib"] = gpu_mib()
    if args.compare:
        before = json.loads(args.compare.read_text())
        report["baseline_mismatches"] = [name for name, value in report["cases"].items()
            if value["sha256"] != before["cases"][name]["sha256"] or
               value["status"] != before["cases"][name]["status"]]
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    for name, value in report["cases"].items():
        expected = 400 if name == "invalid-image" else 200
        assert value["status"] == expected, (name, value["status"])
    assert not report["concurrent_mismatches"]
    assert not report.get("baseline_mismatches")


if __name__ == "__main__":
    main()
