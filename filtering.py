from typing import Union, List
from math import ceil

from PIL import Image
import numpy as np
from tqdm import trange


def logical_filter(binary: Image.Image):
    np_img = np.asarray(binary)
    np_img_result = np_img.copy()
    h, w = np_img.shape
    for x in trange(w):
        for y in range(h):
            all_zeros = True
            all_ones = True
            for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if 0 <= x + dx < w and 0 <= y + dy < h:
                    if np_img[y + dy, x + dx] == 0:
                        all_ones = False
                    else:
                        all_zeros = False
            if all_zeros:
                np_img_result[y, x] = 0
            elif all_ones:
                np_img_result[y, x] = 255
    return Image.fromarray(np_img_result)


def erosion(grayscale: Image.Image, window: Union[np.ndarray, List[List]]):
    if not isinstance(window, np.ndarray):
        window = np.array(window)
    np_img = np.asarray(grayscale)
    np_img_result = np.empty_like(np_img)
    h, w = np_img.shape
    ww, hw = window.shape
    y_offset = ceil(hw / 2)
    x_offset = ceil(ww / 2)
    for x in trange(w, desc='erosion'):
        for y in range(h):
            min_value = 255
            for dx in range(ww):
                for dy in range(hw):
                    if window[dx, dy] == 0:
                        continue
                    y1 = y + dy - y_offset
                    x1 = x + dx - x_offset
                    if 0 <= y1 < h and 0 <= x1 < w and min_value > np_img[y1, x1]:
                        min_value = np_img[y1, x1]
            np_img_result[y, x] = min_value
    return Image.fromarray(np_img_result)
