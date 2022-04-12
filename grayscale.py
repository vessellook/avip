from typing import Literal

from PIL import Image
import numpy as np


def transform(img: Image.Image, method: Literal['max', 'min', 'average'] = 'average') -> Image.Image:
    np_img = np.asarray(img)
    if len(np_img.shape) == 2:
        # img is grayscale
        return img
    if method == 'average':
        np_img_result = np.floor(np_img.sum(axis=2) / 3).astype(np.uint8)
    elif method == 'max':
        np_img_result = np_img.max(axis=2, initial=0)
    elif method == 'min':
        np_img_result = np_img.min(axis=2, initial=0)
    else:
        raise AttributeError(f'Unknown method {method}')
    return Image.fromarray(np_img_result)


def difference(img1: Image.Image, img2: Image.Image) -> Image.Image:
    np_img1 = np.asarray(img1)
    np_img2 = np.asarray(img2)
    np_img_result = np.abs(np_img1 - np_img2)
    return Image.fromarray(np_img_result)


def integral(np_img: np.ndarray) -> np.ndarray:
    return np.cumsum(np.cumsum(np_img, axis=0), axis=1)
