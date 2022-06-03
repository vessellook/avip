from pathlib import Path

import numpy as np
from PIL import Image
import cv2 as cv
from tqdm import trange

from thresholding import transform_bernsen, transform_niblack, transform_otsu, balanced_thresholding
from metrics import cpm

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
    print(prefix, f'cpm == {cpm(image, expected)}, weight(image) == {image.sum()}')


def main():
    convert_to_grayscale()
    output_dir = Path(__file__).parent.joinpath('images/output')
    output_dir.mkdir(exist_ok=True)
    best_size = None
    best_k = None
    best_cpm = 1000
    # for file in list(sorted(expected_dir.iterdir()))[:1]:
    #     expected: np.ndarray = cv.imread(str(file), cv.IMREAD_GRAYSCALE) / 255
    #     grayscale: np.ndarray = cv.imread(str(grayscale_dir / file.name), cv.IMREAD_GRAYSCALE)
    #     for size in trange(3, 200, 2):
    #         for k in np.arange(-1, 1, 0.05):
    #             result = transform_niblack(grayscale, size, k)
    #             current_cpm = cpm(result, expected)
    #             if best_cpm > current_cpm:
    #                 best_cpm = current_cpm
    #                 best_k = k
    #                 best_size = size
    # # лучшие -0.7999999999999998 199 0.05494139749327501
    print(best_k, best_size, best_cpm)
    best_k, best_size = -0.8, 199
    for file in expected_dir.iterdir():
        grayscale = cv.imread(str(grayscale_dir / file.name), cv.IMREAD_GRAYSCALE)
        transform_niblack(grayscale, best_size, best_k)


if __name__ == '__main__':
    main()
