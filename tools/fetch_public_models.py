#!/usr/bin/env python3
"""Fetch public PP-OCRv6 medium ONNX assets from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPOS = {
    "det": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
    "rec": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
}
FILES = ["README.md", "inference.yml", "inference.json", "inference.onnx"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/ppocrv6-medium/source"),
        help="Directory that will receive det/ and rec/ subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    metadata = {"models": {}}
    for kind, repo in REPOS.items():
        out = root / kind
        out.mkdir(parents=True, exist_ok=True)
        info = api.model_info(repo)
        metadata["models"][kind] = {
            "repo": repo,
            "sha": info.sha,
            "license": (info.card_data or {}).get("license")
            if isinstance(info.card_data, dict)
            else getattr(info.card_data, "license", None),
            "files": {},
        }
        for filename in FILES:
            path = hf_hub_download(repo, filename=filename, local_dir=out)
            metadata["models"][kind]["files"][filename] = str(Path(path))
            print(f"{repo}\t{filename}\t{path}")
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"metadata\t{metadata_path}")


if __name__ == "__main__":
    main()
