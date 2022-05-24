import numpy as np
from matplotlib import pyplot as plt


def inside(np_image, x, y):
    return 0 <= x < np_image.shape[1] and 0 <= y < np_image.shape[0]


def get_matrix(np_image: np.ndarray, d, diag=False):
    matrix = np.zeros((256, 256))

    def add(x1, y1, x2, y2):
        if inside(np_image, x1, y1) and inside(np_image, x2, y2):
            matrix[np_image[y1, x1], np_image[y2, x2]] += 1

    for y in range(np_image.shape[0]):
        for x in range(np_image.shape[1]):
            if diag:
                add(x, y, x - d, y - d)
                add(x, y, x - d, y + d)
                add(x, y, x + d, y - d)
                add(x, y, x + d, y + d)
            else:
                add(x, y, x - d, y)
                add(x, y, x + d, y)
                add(x, y, x, y - d)
                add(x, y, x, y + d)
    return matrix


def log_contrast(image: np.ndarray):
    log = np.log2(image + 1)
    log = (256 / np.log2(256)) * log
    return (log - 1).astype(np.uint8)


def corr(matrix: np.ndarray):
    p_j = matrix.sum(axis=0)
    p_i = matrix.sum(axis=1)

    mean_i = (np.arange(1, 257) * p_i).sum()
    mean_j = (np.arange(1, 257) * p_j).sum()

    sigma_i, sigma_j = 0, 0
    for i in range(1, 257):
        sigma_i += (i - mean_j) ** 2 * p_i[i - 1]
        sigma_j += (i - mean_i) ** 2 * p_j[i - 1]

    s = 0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            s += i * j * value

    return (s - mean_i * mean_j) / sigma_i / sigma_j


def save_histogram(array, path):
    hist, bins = np.histogram(array, 256, [0, 255])
    cs = hist.cumsum()
    cs_normalized = cs * float(hist.max()) / cs.max()
    plt.hist(array, bins=255)
    plt.xlim([0, 255])
    plt.plot(cs_normalized)
    plt.legend(('кумулятивная сумма', 'гистограмма'), loc='upper left')
    plt.savefig(str(path))
    plt.clf()
