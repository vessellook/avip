from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters.thresholding import threshold_sauvola


def make_segments(original: Image.Image, gt: Image.Image, size: int):
    w, h = original.width, original.height
    pairs = []
    for x in range(0, w, size):
        for y in range(0, h, size):
            box = x, y, x + size, y + size
            pair = original.crop(box), gt.crop(box)
            pairs.append(pair)
    return pairs


def _key(f):
    return f.name.split('.')[0].split('_')[0]


def collect_images(root: Path):
    pairs = []
    for year_dir in root.iterdir():
        if not year_dir.is_dir():
            continue
        input_dirs = [f for f in year_dir.iterdir() if f.is_dir() and f.name.startswith('original')]
        for input_dir in input_dirs:
            gt_dir = year_dir / input_dir.name.replace('original', 'gt')
            if not gt_dir.exists() or not gt_dir.is_dir():
                continue
            input_images = sorted(input_dir.iterdir(), key=_key)
            gt_images = sorted(gt_dir.iterdir(), key=_key)
            pairs.extend(zip(input_images, gt_images))

    return pairs


def main(input_dir, original_output_dir, gt_output_dir):
    images = collect_images(input_dir)
    counter = 0
    for num, (original, gt) in enumerate(images, 1):
        original = Image.open(str(original))
        gt = Image.open(str(gt))
        segments = make_segments(original, gt, 256)
        for original_segment, gt_segment in segments:
            original_segment.save(str(original_output_dir / f'{counter}.bmp'))
            gt_segment.save(str(gt_output_dir / f'{counter}.bmp'))
            counter += 1
        print(f'{num} / {len(images)}, {counter} segments')


if __name__ == '__main__':
    root = Path(__file__).parent / 'images/dibco/images'
    original_dir = Path(__file__).parent / 'images/dibco/segments/original'
    gt_dir = Path(__file__).parent / 'images/dibco/segments/gt'
    main(root, original_dir, gt_dir)
