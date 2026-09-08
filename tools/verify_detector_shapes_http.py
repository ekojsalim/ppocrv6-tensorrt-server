#!/usr/bin/env python3
"""Check small/extreme page shapes and coordinates against explicit white padding."""
import argparse
import json
from pathlib import Path
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from verify_workspace_http import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8184')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    def recognize(im):
        req = urllib.request.Request(args.base_url + '/v1/ocr/recognize',
            data=json.dumps({'image': encode(im)}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    sizes = [(1, 1), (1, 4000), (4000, 1), (128, 128), (192, 640),
        (640, 192), (640, 224), (640, 239), (640, 240), (640, 256),
        (256, 256), (1280, 256), (2560, 256), (256, 2560),
        (3200, 480), (3200, 600), (4000, 750), (4000, 4000)]
    for size in sizes:
        result = recognize(Image.new('RGB', size, 'white'))
        assert not result['lines'], (size, result)

    font_path = '/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf'
    checks = []
    for w, h, pw, ph, font_size in [(640, 192, 640, 256, 32),
            (192, 640, 256, 640, 24), (128, 128, 256, 256, 24),
            (2560, 256, 2560, 512, 64), (256, 2560, 512, 2560, 32)]:
        im = Image.new('RGB', (w, h), 'white')
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font_path, font_size)
        draw.text((16, 24), 'OCR 123', font=font, fill='black')
        canvas = Image.new('RGB', (pw, ph), 'white')
        canvas.paste(im, (0, 0))
        actual, control = recognize(im), recognize(canvas)
        assert actual['lines'] and control['lines'], (w, h)
        assert len(actual['lines']) == len(control['lines']), (w, h)
        for a, b in zip(actual['lines'], control['lines']):
            assert a['text'] == b['text'], (w, h, 'text')
            assert a['box'] == b['box'], (w, h, 'coordinates', a['box'], b['box'])
            for x, y in a['box']:
                assert 0 <= x <= w and 0 <= y <= h, (w, h, x, y)
        checks.append({'width': w, 'height': h, 'lines': len(actual['lines']),
                       'text_and_coordinates_match_explicit_padding': True})
    report = {'blank_shapes_passed': sizes, 'coordinate_checks': checks}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
