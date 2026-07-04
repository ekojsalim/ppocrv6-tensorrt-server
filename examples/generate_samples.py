#!/usr/bin/env python3
"""Generate tiny PPM fixtures for smoke-testing the HTTP server."""

from __future__ import annotations

from pathlib import Path


FONT: dict[str, list[str]] = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
}


def blank(width: int, height: int) -> list[list[int]]:
    return [[255 for _ in range(width)] for _ in range(height)]


def draw_text(
    pixels: list[list[int]],
    text: str,
    *,
    x: int,
    y: int,
    scale: int,
) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char)
        if glyph is None:
            glyph = FONT[" "]
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        px = cursor + gx * scale + dx
                        py = y + gy * scale + dy
                        if 0 <= py < len(pixels) and 0 <= px < len(pixels[0]):
                            pixels[py][px] = 0
        cursor += 6 * scale


def write_ppm(path: Path, pixels: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height = len(pixels)
    width = len(pixels[0])
    body = bytearray()
    for row in pixels:
        for value in row:
            body.extend((value, value, value))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(body))


def generate_samples(out_dir: Path) -> list[Path]:
    glyph = blank(48, 48)
    draw_text(glyph, "A", x=6, y=4, scale=6)
    write_ppm(out_dir / "glyph_A.ppm", glyph)

    page = blank(1000, 1300)
    draw_text(page, "HELLO OCR", x=80, y=160, scale=14)
    draw_text(page, "TEST 123", x=80, y=360, scale=14)
    draw_text(page, "PAGE 01", x=80, y=560, scale=14)
    write_ppm(out_dir / "page_ocr_portrait.ppm", page)

    landscape = blank(1300, 1000)
    draw_text(landscape, "HELLO OCR", x=80, y=120, scale=14)
    draw_text(landscape, "TEST 123", x=80, y=320, scale=14)
    draw_text(landscape, "PAGE 01", x=80, y=520, scale=14)
    write_ppm(out_dir / "page_ocr_landscape.ppm", landscape)

    return [
        out_dir / "glyph_A.ppm",
        out_dir / "page_ocr_portrait.ppm",
        out_dir / "page_ocr_landscape.ppm",
    ]


def main() -> None:
    out_dir = Path("examples/samples")
    for path in generate_samples(out_dir):
        print(path)


if __name__ == "__main__":
    main()
