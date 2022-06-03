import numpy as np


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def cpm_simple(real: np.ndarray, expected: np.ndarray):
    return abs(real.sum(dtype=np.float) - expected.sum(dtype=np.float))


def cpm(real: np.ndarray, expected: np.ndarray, size: int = 30):
    """count pseudometric measure - сравнение результатов бинаризации на основе количества чёрного

    источник: http://www.isa.ru/proceedings/images/documents/2013-63-3/t-3-13_85-94.pdf
    """
    if not isinstance(real, np.ndarray):
        real = np.asarray(real) / 255
    if not isinstance(expected, np.ndarray):
        expected = np.asarray(expected) / 255
    h, w = real.shape
    real_list = blockshaped(real[h % size:, w % size:], h // size, w // size)
    expected_list = blockshaped(expected[h % size:, w % size:], h // size, w // size)
    result_sum = sum(cpm_simple(a, b) for a, b in zip(real_list, expected_list))
    return result_sum / expected.sum()
