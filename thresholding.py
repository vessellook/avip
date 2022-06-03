from math import floor, ceil
from typing import Literal, Union, List, Tuple

import numpy as np
from PIL import Image
from tqdm import trange

from grayscale import integral


def balanced_thresholding(grayscale: Union[Image.Image, np.ndarray]) -> np.ndarray:
    np_img = np.asarray(grayscale, dtype=np.uint8)
    min_val = np_img.min()
    max_val = np_img.max() + 1
    hist, edges = np.histogram(np_img, density=True, bins=np.arange(min_val, max_val))
    left = 0
    right = len(hist) - 1
    while left < right:
        left_middle = floor((left + right) / 2)
        right_middle = ceil((left + right) / 2)
        left_sum = np.sum(hist[left:left_middle])
        right_sum = np.sum(hist[right_middle:right])
        if left_sum > right_sum:
            left += 1
        else:
            right -= 1
    binary = np.zeros_like(grayscale)
    binary[grayscale > edges[left]] = 1
    return binary


def _otsu_convertation(p0, p1, i0, i1, w0, w1, metric: Literal['inter', 'intra', 'div'] = 'div'):
    m0 = np.sum(p0 * i0 / w0)
    m1 = np.sum(p1 * i1 / w1)
    d0 = np.sum(p0 * (i0 - m0) ** 2)
    d1 = np.sum(p1 * (i1 - m1) ** 2)
    d_intra = w0 * d0 + w1 * d1
    d_inter = w0 * w1 * (m0 - m1) ** 2
    if metric == 'inter':
        return d_inter
    elif metric == 'intra':
        return d_intra
    else:
        return d_inter / d_intra


def transform_otsu(grayscale: Union[Image.Image, np.ndarray], with_tqdm=False) -> np.ndarray:
    np_img = np.asarray(grayscale, dtype=np.uint8)
    min_val = np_img.min()
    max_val = np_img.max() + 1
    hist, edges = np.histogram(np_img, density=True, bins=np.arange(min_val, max_val))

    w0 = np.cumsum(hist)
    w1 = np.ones_like(w0) - w0
    results = []
    if with_tqdm:
        iterator = trange(len(w0), desc='otsu')
    else:
        iterator = range(len(w0))
    for i in iterator:
        if w0[i] < 0.01 or w1[i] < 0.01:
            continue
        result = _otsu_convertation(
            p0=hist[:i], i0=edges[1:i + 1], w0=w0[i],
            p1=hist[i:], i1=edges[i + 1:], w1=w1[i],
        )
        results.append(result)
    threshold = edges[results.index(max(results)) + 1]
    binary = np.zeros_like(grayscale)
    binary[grayscale > threshold] = 1
    return binary


def transform_bernsen(grayscale: Union[Image.Image, np.ndarray], s: int, t: float = 0) -> np.ndarray:
    np_image = np.asarray(grayscale)
    binary = np.zeros_like(np_image)
    np_integral = integral(np_image)
    h, w = np_image.shape
    for y, row in enumerate(np_image):
        for x, value in enumerate(row):
            top = max(0, y - s)
            bottom = min(h - 1, y + s)
            left = max(0, x - s)
            right = min(w - 1, x + s)
            top_left = np_integral[top, left]
            top_right = np_integral[top, right]
            bottom_left = np_integral[bottom, left]
            bottom_right = np_integral[bottom, right]
            average_value = top_left + bottom_right - top_right - bottom_left

            if value > average_value * t:
                binary[y, x] = 1
    return binary


def transform_aquil(grayscale: Image.Image, r: int = 3, R: int = 15):
    pass


def _get_image_and_submask(image: np.ndarray, x: int, y: int, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    c_x = mask.shape[1] // 2
    c_y = mask.shape[0] // 2
    left = min(c_x, x)
    right = min(c_x, w - x)
    top = min(c_y, y)
    bottom = min(c_y, h - y)
    return image[y - top: y + bottom, x - left: x + right], mask[c_y - top: c_y + bottom, c_x - left: c_x + right]


def transform_niblack(grayscale: Image.Image, mask: Union[np.ndarray, List[List]], k: float = -0.2,
                      global_min_threshold: int = 0, global_max_threshold: int = 255) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    np_image = np.asarray(grayscale)
    binary = np.zeros_like(np_image)
    # mean = np.zeros(np_image)
    # stdev = np.zeros(np_image)
    for y, row in enumerate(np_image):
        for x, value in enumerate(row):
            if value < global_min_threshold:
                continue
            if value > global_max_threshold:
                binary[y, x] = 1
                continue
            window, submask = _get_image_and_submask(np_image, x, y, mask)
            mean = (window * submask).sum() / submask.sum()
            stdev = (window ** 2 * submask).sum() / submask.sum() - mean ** 2
            threshold = mean + k * stdev
            if threshold < value:
                binary[y, x] = 1
            # mean[y, x] = (window * submask).sum() / submask.sum()
            # stdev[y, x] = (window ** 2 * submask).sum() / submask.sum() - mean[y, x] ** 2
    return binary
