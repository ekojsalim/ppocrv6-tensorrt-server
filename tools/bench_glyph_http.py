#!/usr/bin/env python3
"""Benchmark the PP-OCRv6 glyph HTTP API.

The script sends single or batched base64 image payloads to
`/v1/glyphs/recognize` and reports latency/throughput. If no image directory is
provided, it uses a generated 48x48 PPM image. That synthetic image is useful
for a dependency-free smoke/microbenchmark only. Use representative PNG or
JPEG inputs for production throughput measurements.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".ppm", ".pgm"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_csv_ints(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return items


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def synthetic_ppm() -> bytes:
    width = height = 48
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            ink = 12 <= x < 36 and (10 <= y < 16 or 22 <= y < 28 or 34 <= y < 40)
            pixels.extend((0, 0, 0) if ink else (255, 255, 255))
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(pixels)


def encode_image_bytes(raw: bytes, mime: str) -> str:
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_images(
    glyph_dir: Path | None, image_path: Path | None, unique_count: int | None
) -> list[str]:
    if image_path is not None:
        raws = [image_path.read_bytes()]
        if unique_count is not None:
            raws = make_unique_png_payloads(raws, unique_count)
        return [encode_image_bytes(raw, "application/octet-stream") for raw in raws]

    if glyph_dir is not None:
        paths = sorted(
            path
            for path in glyph_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise SystemExit(f"No image files found in {glyph_dir}")
        raws = [path.read_bytes() for path in paths]
        if unique_count is not None:
            raws = make_unique_png_payloads(raws, unique_count)
        return [encode_image_bytes(raw, "application/octet-stream") for raw in raws]

    return [encode_image_bytes(synthetic_ppm(), "image/x-portable-pixmap")]


def make_unique_png_payloads(raws: list[bytes], count: int) -> list[bytes]:
    if not raws:
        raise SystemExit("cannot create unique payloads from an empty image list")
    if any(not raw.startswith(PNG_SIGNATURE) for raw in raws):
        raise SystemExit("--unique-image-payloads currently requires PNG input images")
    return [add_png_text_chunk(raws[index % len(raws)], index) for index in range(count)]


def add_png_text_chunk(raw: bytes, index: int) -> bytes:
    iend_offset = find_png_iend_offset(raw)
    chunk_data = b"bench_id\x00" + str(index).encode("ascii")
    crc = binascii.crc32(b"tEXt")
    crc = binascii.crc32(chunk_data, crc) & 0xFFFFFFFF
    chunk = (
        struct.pack(">I", len(chunk_data))
        + b"tEXt"
        + chunk_data
        + struct.pack(">I", crc)
    )
    return raw[:iend_offset] + chunk + raw[iend_offset:]


def find_png_iend_offset(raw: bytes) -> int:
    offset = len(PNG_SIGNATURE)
    while offset + 12 <= len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        next_offset = offset + 12 + length
        if next_offset > len(raw):
            break
        if chunk_type == b"IEND":
            return offset
        offset = next_offset
    raise SystemExit("PNG input is missing an IEND chunk")


def make_image_payload(
    images: list[str],
    count: int,
    width: int,
    batch_size: int,
    character_policy: str,
) -> dict[str, Any]:
    selected = [images[i % len(images)] for i in range(count)]
    return {
        "images": selected,
        "width": width,
        "batch_size": batch_size,
        "character_policy": character_policy,
    }


def post_json(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], int, int]:
    raw = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=raw,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return json.loads(body.decode("utf-8")), len(body), len(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {body}") from exc


def run_case(
    *,
    url: str,
    images: list[str],
    count: int,
    width: int,
    batch_size: int,
    character_policy: str,
    warmups: int,
    repeats: int,
    timeout: float,
) -> dict[str, Any]:
    payload = make_image_payload(images, count, width, batch_size, character_policy)
    for _ in range(warmups):
        post_json(url, payload, timeout)

    latencies_ms: list[float] = []
    response_bytes = 0
    request_bytes = 0
    server_predict_ms: list[float] = []
    server_glyphs_per_second: list[float] = []

    for _ in range(repeats):
        start = time.perf_counter()
        data, response_bytes, request_bytes = post_json(url, payload, timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        meta = data.get("meta", {})
        if isinstance(meta.get("predict_elapsed_ms"), (int, float)):
            server_predict_ms.append(float(meta["predict_elapsed_ms"]))
        if isinstance(meta.get("glyphs_per_second"), (int, float)):
            server_glyphs_per_second.append(float(meta["glyphs_per_second"]))

    median_ms = statistics.median(latencies_ms)
    p95_ms = percentile(latencies_ms, 0.95)
    mean_ms = statistics.fmean(latencies_ms)
    return {
        "count": count,
        "width": width,
        "batch_size": batch_size,
        "repeats": repeats,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "client_glyphs_per_s": count / (median_ms / 1000.0),
        "server_predict_median_ms": statistics.median(server_predict_ms)
        if server_predict_ms
        else None,
        "server_glyphs_per_s_median": statistics.median(server_glyphs_per_second)
        if server_glyphs_per_second
        else None,
        "response_bytes": response_bytes,
        "request_bytes": request_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8184/v1/glyphs/recognize",
        help="Glyph recognition endpoint.",
    )
    parser.add_argument("--glyph-dir", type=Path, help="Directory of glyph images to cycle through.")
    parser.add_argument("--image", type=Path, help="Single image file to send repeatedly.")
    parser.add_argument(
        "--unique-image-payloads",
        action="store_true",
        help="Make repeated PNG payloads byte-unique to avoid server cache hits.",
    )
    parser.add_argument("--counts", type=parse_csv_ints, default=parse_csv_ints("1,6,16,32,64"))
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--character-policy",
        choices=["cjk_focus", "cjk_focus_fallback", "suppress_ascii", "all"],
        default="cjk_focus_fallback",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json", action="store_true", help="Emit JSON lines instead of TSV.")
    args = parser.parse_args()

    if args.width <= 0 or args.batch_size <= 0:
        raise SystemExit("--width and --batch-size must be positive")

    unique_count = max(args.counts) if args.unique_image_payloads else None
    images = load_images(args.glyph_dir, args.image, unique_count)
    if not args.json:
        print(
            "\t".join(
                [
                    "count",
                    "unique_payloads",
                    "width",
                    "batch_size",
                    "median_ms",
                    "p95_ms",
                    "client_glyphs_per_s",
                    "server_predict_median_ms",
                    "server_glyphs_per_s_median",
                    "request_bytes",
                    "response_bytes",
                ]
            )
        )

    for count in args.counts:
        result = run_case(
            url=args.url,
            images=images,
            count=count,
            width=args.width,
            batch_size=args.batch_size,
            character_policy=args.character_policy,
            warmups=args.warmups,
            repeats=args.repeats,
            timeout=args.timeout,
        )
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(
                "\t".join(
                    [
                        str(result["count"]),
                        str(args.unique_image_payloads).lower(),
                        str(result["width"]),
                        str(result["batch_size"]),
                        f"{result['median_ms']:.3f}",
                        f"{result['p95_ms']:.3f}",
                        f"{result['client_glyphs_per_s']:.1f}",
                        ""
                        if result["server_predict_median_ms"] is None
                        else f"{result['server_predict_median_ms']:.3f}",
                        ""
                        if result["server_glyphs_per_s_median"] is None
                        else f"{result['server_glyphs_per_s_median']:.1f}",
                        str(result["request_bytes"]),
                        str(result["response_bytes"]),
                    ]
                )
            )


if __name__ == "__main__":
    main()
