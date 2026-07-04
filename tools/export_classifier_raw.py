#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PP-OCRv6 classifier NPZ to raw FP16 files.")
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("artifacts/ppocrv6-medium/derived/classifier.npz"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("artifacts/ppocrv6-medium/source/rec/inference.yml"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/ppocrv6-medium/classifier"),
    )
    return parser.parse_args()


def read_character_list(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    chars = config["PostProcess"]["character_dict"]
    return ["blank", *[str(char) for char in chars], " "]


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.classifier)
    weight = np.ascontiguousarray(data["weight"].astype(np.float16, copy=False))
    bias = np.ascontiguousarray(data["bias"].astype(np.float16, copy=False))
    weight_path = args.out_dir / "weight.fp16.bin"
    bias_path = args.out_dir / "bias.fp16.bin"
    characters_path = args.out_dir / "characters.txt"
    meta_path = args.out_dir / "metadata.json"
    weight.tofile(weight_path)
    bias.tofile(bias_path)
    characters = read_character_list(args.config)
    with characters_path.open("w", encoding="utf-8") as f:
        for char in characters:
            f.write(char)
            f.write("\n")
    metadata = {
        "classifier": str(args.classifier),
        "config": str(args.config),
        "weight_path": str(weight_path),
        "bias_path": str(bias_path),
        "characters_path": str(characters_path),
        "character_count": len(characters),
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "weight_dtype": str(weight.dtype),
        "bias_dtype": str(bias.dtype),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
