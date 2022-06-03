import numpy as np


def integral(np_img: np.ndarray) -> np.ndarray:
    return np.cumsum(np.cumsum(np_img, axis=0), axis=1)
