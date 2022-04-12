from math import ceil
from typing import Tuple

import numpy as np


def get_partial_x(np_img: np.ndarray) -> np.ndarray:
    return np_img[:, 1:] - np_img[:, :-1]


def get_partial_y(np_img: np.ndarray) -> np.ndarray:
    return np_img[1:, :] - np_img[:-1, :]


def get_gradient_euclid(partial_x: np.ndarray, partial_y: np.ndarray) -> np.ndarray:
    return np.sqrt(partial_x ** 2 + partial_y ** 2)


def get_gradient_abs(partial_x: np.ndarray, partial_y: np.ndarray) -> np.ndarray:
    return np.abs(partial_x) + np.abs(partial_y)


def normalize(gradient: np.ndarray) -> np.ndarray:
    return (255 * gradient / np.max(gradient)).astype(np.uint8)


def get_partial(np_img: np.ndarray, window: np.ndarray) -> np.ndarray:
    h, w = window.shape
    x_shift = ceil(w / 2)
    y_shift = ceil(h / 2)
    result = np.zeros_like(np_img[h - 1:, w - 1:])
    for y in range(h):
        for x in range(w):
            y1 = -h + y + 1
            x1 = -w + x + 1
            if y1 and x1:
                result += window[y, x] * np_img[y: -h + y + 1, x: -w + x + 1]
            elif y1:
                result += window[y, x] * np_img[y: -h + y + 1, x:]
            elif x1:
                result += window[y, x] * np_img[y:, x: -w + x + 1]
            else:
                result += window[y, x] * np_img[y:, x:]
    return result


def get_pruitt(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param n: odd number
    :return: tuple of two Pruitt operator matrices
    """
    line = np.zeros((n, 1), dtype=np.int8)
    line[0] = -1
    line[-1] = 1
    x = np.repeat(line, n, axis=1)
    return x, x.T
