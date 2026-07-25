#!/usr/bin/env python3
"""Post a sample image to a running PP-OCRv6 TensorRT server."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8184")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--kind", choices=["glyph", "ocr"], required=True)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument(
        "--character-policy",
        choices=["cjk_focus", "cjk_focus_fallback", "suppress_ascii", "all"],
        help="Glyph vocabulary policy; omitted uses the server default.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON response.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
    if args.kind == "glyph":
        url = f"{args.base_url}/v1/glyphs/recognize"
        payload = {"image": encoded, "width": args.width}
        if args.character_policy is not None:
            payload["character_policy"] = args.character_policy
    else:
        url = f"{args.base_url}/v1/ocr/recognize"
        payload = {"image": encoded}

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.kind == "glyph":
        prediction = result.get("prediction") or (result.get("predictions") or [{}])[0]
        print(json.dumps(prediction, ensure_ascii=False))
    else:
        print(f"count={result.get('count')}")
        for line in result.get("lines", [])[:8]:
            print(f"{line.get('index')}: {line.get('text')} score={line.get('score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
