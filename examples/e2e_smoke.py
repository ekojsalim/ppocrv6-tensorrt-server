#!/usr/bin/env python3
"""End-to-end HTTP smoke checks for the PP-OCRv6 TensorRT server."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
sys.path.insert(0, str(EXAMPLES_DIR))

from generate_samples import generate_samples  # noqa: E402


class SmokeFailure(RuntimeError):
    pass


def parse_expected_shape(value: str) -> list[int] | None:
    if value.lower() in {"any", "none", "skip", ""}:
        return None
    try:
        shape = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--expected-detector-shape must be comma-separated ints") from exc
    if len(shape) != 4:
        raise argparse.ArgumentTypeError("--expected-detector-shape must contain 4 dimensions")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8184")
    parser.add_argument("--samples-dir", type=Path, default=REPO_ROOT / "examples" / "samples")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--glyph-width", type=int, default=80)
    parser.add_argument("--glyph-min-score", type=float, default=0.5)
    parser.add_argument("--ocr-min-count", type=int, default=3)
    parser.add_argument(
        "--expected-detector-shape",
        type=parse_expected_shape,
        default=[1, 3, 1280, 992],
        help="Comma-separated expected portrait OCR detector shape, or 'any' to skip.",
    )
    parser.add_argument(
        "--expected-landscape-detector-shape",
        type=parse_expected_shape,
        default=[1, 3, 992, 1280],
        help="Comma-separated expected landscape OCR detector shape, or 'any' to skip.",
    )
    parser.add_argument(
        "--require-page-token",
        action="append",
        default=None,
        help="Normalized token/text expected somewhere in the page OCR output.",
    )
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-landscape", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional directory for full HTTP JSON responses.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Require existing sample files instead of generating them.",
    )
    args = parser.parse_args()
    if args.require_page_token is None:
        args.require_page_token = ["HELLO OCR", "TEST", "123"]
    return args


def request_json(base_url: str, path: str, timeout: float, payload: dict[str, Any] | None = None) -> Any:
    url = base_url.rstrip("/") + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SmokeFailure(f"{path} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SmokeFailure(f"{path} request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SmokeFailure(f"{path} timed out after {timeout:g}s") from exc


def image_payload(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def score(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return 0.0


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.upper()).strip()


def write_json(out_dir: Path | None, name: str, payload: Any) -> None:
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def ensure_samples(samples_dir: Path, no_generate: bool) -> tuple[Path, Path, Path]:
    glyph = samples_dir / "glyph_A.ppm"
    page = samples_dir / "page_ocr_portrait.ppm"
    landscape = samples_dir / "page_ocr_landscape.ppm"
    if not no_generate and (not glyph.exists() or not page.exists() or not landscape.exists()):
        generate_samples(samples_dir)
    expect(glyph.is_file(), f"missing glyph sample: {glyph}")
    expect(page.is_file(), f"missing page sample: {page}")
    expect(landscape.is_file(), f"missing landscape page sample: {landscape}")
    return glyph, page, landscape


def check_health(args: argparse.Namespace) -> None:
    health = request_json(args.base_url, "/health", args.timeout)
    expect(health.get("ok") is True, f"unexpected health response: {health!r}")
    print("PASS health")


def check_glyph(args: argparse.Namespace, glyph_path: Path) -> None:
    result = request_json(
        args.base_url,
        "/v1/glyphs/recognize",
        args.timeout,
        {
            "image": image_payload(glyph_path),
            "width": args.glyph_width,
            "character_policy": "all",
        },
    )
    write_json(args.json_out, "glyph-response.json", result)
    prediction = result.get("prediction")
    if prediction is None:
        predictions = result.get("predictions") or []
        prediction = predictions[0] if predictions else {}
    text = prediction.get("text")
    prediction_score = score(prediction.get("score"))
    expect(
        result.get("character_policy") == "all",
        "glyph smoke test did not opt out of ASCII suppression",
    )
    expect(text == "A", f"glyph fixture decoded as {text!r}, expected 'A'")
    expect(
        prediction_score >= args.glyph_min_score,
        f"glyph score {prediction_score:.4f} < {args.glyph_min_score:.4f}",
    )
    print(f"PASS glyph text={text!r} score={prediction_score:.4f}")


def check_ocr(
    args: argparse.Namespace,
    page_path: Path,
    *,
    label: str,
    expected_shape: list[int] | None,
) -> None:
    result = request_json(
        args.base_url,
        "/v1/ocr/recognize",
        args.timeout,
        {"image": image_payload(page_path)},
    )
    write_json(args.json_out, f"ocr-{label}-response.json", result)
    count = int(result.get("count") or 0)
    lines = result.get("lines") or []
    expect(count >= args.ocr_min_count, f"OCR line count {count} < {args.ocr_min_count}")
    expect(len(lines) == count, f"OCR count {count} does not match lines length {len(lines)}")
    if expected_shape is not None:
        actual_shape = result.get("detector_input_shape")
        expect(
            actual_shape == expected_shape,
            f"detector_input_shape {actual_shape!r} != {expected_shape!r}",
        )

    line_texts = [normalize_text(str(line.get("text") or "")) for line in lines]
    joined = normalize_text(" ".join(line_texts))
    for token in args.require_page_token:
        normalized_token = normalize_text(token)
        expect(normalized_token in joined, f"OCR output missing token {token!r}: {line_texts!r}")

    preview = ", ".join(repr(text) for text in line_texts[:6])
    print(
        f"PASS ocr-{label} count={count} "
        f"detector_shape={result.get('detector_input_shape')} lines=[{preview}]"
    )


def main() -> int:
    args = parse_args()
    try:
        glyph_path, page_path, landscape_path = ensure_samples(args.samples_dir, args.no_generate)
        check_health(args)
        check_glyph(args, glyph_path)
        if not args.skip_ocr:
            check_ocr(
                args,
                page_path,
                label="portrait",
                expected_shape=args.expected_detector_shape,
            )
            if not args.skip_landscape:
                check_ocr(
                    args,
                    landscape_path,
                    label="landscape",
                    expected_shape=args.expected_landscape_detector_shape,
                )
    except SmokeFailure as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
