from fractions import Fraction
from math import floor
from typing import Union

from PIL import Image
import numpy as np
from tqdm import tqdm, trange


def interpolation(img: Image.Image, n: int) -> Image.Image:
    np_img = np.asarray(img)
    if len(np_img.shape) == 3:
        h, w, colors = np_img.shape
        h1 = h * n
        w1 = w * n
        np_img_result = np.empty((h1, w1, colors), dtype=np.uint8)
    else:
        h, w = np_img.shape
        h1 = h * n
        w1 = w * n
        np_img_result = np.empty((h1, w1), dtype=np.uint8)

    for y1, y in tqdm(zip(range(h1), np.repeat(range(h), n)), desc='interpolation', total=h1):
        for x1, x in zip(range(w1), np.repeat(range(w), n)):
            np_img_result[y1, x1] = np_img[y, x]
    return Image.fromarray(np_img_result)


def decimation(img: Image.Image, n: int) -> Image.Image:
    np_img = np.asarray(img)
    if len(np_img.shape) == 3:
        h, w, colors = np_img.shape
        h1 = floor(Fraction(h, n))
        w1 = floor(Fraction(w, n))
        np_img_result = np.empty((h1, w1, colors), dtype=np.uint8)
    else:
        h, w = np_img.shape
        h1 = floor(Fraction(h, n))
        w1 = floor(Fraction(w, n))
        np_img_result = np.empty((h1, w1), dtype=np.uint8)
    for y1 in trange(h1, desc='decimation'):
        y = y1 * n
        for x1 in range(w1):
            x = x1 * n
            np_img_result[y1, x1] = np_img[y, x]
    return Image.fromarray(np_img_result)


def resampling_single_pass(img: Image.Image, k: Union[int, Fraction]) -> Image.Image:
    np_img = np.asarray(img)
    if len(np_img.shape) == 3:
        h, w, colors = np_img.shape
        h1 = floor(h * k)
        w1 = floor(w * k)
        np_img_result = np.empty((h1, w1, colors), dtype=np.uint8)
    else:
        h, w = np_img.shape
        h1 = floor(h * k)
        w1 = floor(w * k)
        np_img_result = np.empty((h1, w1), dtype=np.uint8)
    for y1 in trange(h1, desc='single_pass'):
        y = floor(Fraction(y1, k))
        for x1 in range(w1):
            x = floor(Fraction(x1, k))
            np_img_result[y1, x1] = np_img[y, x]
    return Image.fromarray(np_img_result)
