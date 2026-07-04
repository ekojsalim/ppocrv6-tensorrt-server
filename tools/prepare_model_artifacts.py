#!/usr/bin/env python3
"""Prepare reproducible PP-OCRv6 medium artifacts for the native runtime."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/ppocrv6-medium"),
        help="Artifact root directory.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse existing source ONNX/config files instead of downloading them.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    tools_dir = Path(__file__).resolve().parent
    source_dir = args.root / "source"
    derived_dir = args.root / "derived"
    classifier_dir = args.root / "classifier"

    if not args.skip_fetch:
        run(
            [
                sys.executable,
                str(tools_dir / "fetch_public_models.py"),
                "--out-dir",
                str(source_dir),
            ]
        )

    run(
        [
            sys.executable,
            str(tools_dir / "extract_ppocrv6_classifier.py"),
            "--model",
            str(source_dir / "rec" / "inference.onnx"),
            "--output",
            str(derived_dir / "classifier.npz"),
            "--hidden-output",
            str(derived_dir / "rec-hidden.onnx"),
        ]
    )
    run(
        [
            sys.executable,
            str(tools_dir / "export_classifier_raw.py"),
            "--classifier",
            str(derived_dir / "classifier.npz"),
            "--config",
            str(source_dir / "rec" / "inference.yml"),
            "--out-dir",
            str(classifier_dir),
        ]
    )

    print(f"prepared artifacts under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
