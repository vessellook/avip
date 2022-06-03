from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

from features import weight
from thresholding import transform_bernsen, transform_niblack, transform_otsu, balanced_thresholding
from uir import cpm

original_dir = Path(__file__).parent.joinpath('images/training')
grayscale_dir = Path(__file__).parent.joinpath('images/training_grayscale')
grayscale_dir.mkdir(exist_ok=True)
expected_dir = Path(__file__).parent.joinpath('images/training_manual-processing')


def convert_to_grayscale():
    for path in original_dir.iterdir():
        output_path = grayscale_dir / path.name
        if output_path.exists():
            continue
        Image.open(str(path)).convert('L').save(str(output_path))


def compare_and_save(prefix, image, expected, path):
    Image.fromarray(image * 255).save(str(path))
    print(prefix, f'cpm == {cpm(image, expected)}, weight(image) == {weight(image)}')


def main():
    convert_to_grayscale()
    output_dir = Path(__file__).parent.joinpath('images/output')
    output_dir.mkdir(exist_ok=True)

    niblack_mask = np.ones((199, 199))
    for file in expected_dir.iterdir():
        expected = Image.open(str(file)).convert('L')
        grayscale = Image.open(str(grayscale_dir / file.name)).convert('L')
        print('start', file.name, weight(np.asarray(expected) / 255))

        compare_and_save('niblack', transform_niblack(grayscale, niblack_mask, -0.8),
                         expected, output_dir / 'niblack' / file.name)
        compare_and_save('balancing', balanced_thresholding(grayscale),
                         expected, output_dir / 'balancing' / file.name)
        compare_and_save('otsu', transform_otsu(grayscale),
                         expected, output_dir / 'otsu' / file.name)
        print('finish', file.name)


if __name__ == '__main__':
    main()
