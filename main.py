import math
from pathlib import Path

import numpy as np
from PIL import Image
import cv2 as cv
from cv2.ximgproc import niBlackThreshold
from tqdm import trange
from matplotlib import pyplot as plt
from skimage.filters.thresholding import (
    threshold_otsu, threshold_niblack, threshold_sauvola,
)
from skimage.metrics import (
    structural_similarity as ssim,
    mean_squared_error as mse,
    peak_signal_noise_ratio as psnr,
)

from thresholding import transform_bernsen, transform_niblack, transform_otsu, balanced_thresholding
from metrics import cpm, cpm_simple

red_segments_dir = Path(__file__).parent.joinpath('images/red_segments')
original_dir = Path(__file__).parent.joinpath('images/training')
grayscale_dir = Path(__file__).parent.joinpath('images/training_grayscale')
grayscale_dir.mkdir(exist_ok=True)
expected_dir = Path(__file__).parent.joinpath('images/training_manual-processing')


def convert_to_grayscale():
    for path in original_dir.iterdir():
        output_path = grayscale_dir / path.name
        if output_path.exists():
            continue
        channels = Image.open(str(path)).split()
        it = iter(channels)
        next(it)
        next(it)
        next(it).save(str(output_path))


def save(image, path):
    image.dtype = np.uint8
    cv.imwrite(str(path), image * 255)


def compare_and_save(image, expected, path):
    image.dtype = np.uint8
    cv.imwrite(str(path), image * 255)
    print(path)
    print(f'cpm == {cpm(image, expected)}')
    print(f'cpm_simple == {cpm_simple(image, expected)}')
    print(f'mse == {mse(image, expected)}')
    print(f'psnr == {psnr(image, expected)}')
    print(f'ssim == {ssim(image, expected)}')
    print(f'weight(image) == {(1 - image).sum()}')
    print()


def main_old():
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


def save_otsu_niblack_sauvola():
    convert_to_grayscale()
    output_dir = Path(__file__).parent.joinpath('images/output')
    output_dir.mkdir(exist_ok=True)
    for file in list(sorted(expected_dir.iterdir()))[:1]:
        expected: np.ndarray = cv.imread(str(file), cv.IMREAD_GRAYSCALE) // 255
        grayscale: np.ndarray = cv.imread(str(grayscale_dir / file.name), cv.IMREAD_GRAYSCALE)
        otsu = grayscale > threshold_otsu(grayscale)
        compare_and_save(otsu, expected, output_dir / 'otsu' / file.name)
        niblack = grayscale > threshold_niblack(grayscale, window_size=201, k=0.8)
        compare_and_save(niblack, expected, output_dir / 'niblack' / file.name)
        sauvola = grayscale > threshold_sauvola(grayscale, window_size=201, k=0.8)
        compare_and_save(sauvola, expected, output_dir / 'sauvola' / file.name)


def convert_and_save_image(file):
    convert_to_grayscale()
    output_dir = Path(__file__).parent.joinpath('images/output')
    output_dir.mkdir(exist_ok=True)
    grayscale: np.ndarray = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
    otsu = grayscale > threshold_otsu(grayscale)
    save(otsu, output_dir / 'otsu' / file.name)
    niblack = grayscale > threshold_niblack(grayscale, window_size=15, k=1.5)
    save(niblack, output_dir / 'niblack' / file.name)
    sauvola = grayscale > threshold_sauvola(grayscale, window_size=15, k=0.2)
    save(sauvola, output_dir / 'sauvola' / file.name)


def red_segmentation(file):
    image: np.ndarray = cv.imread(str(original_dir / file.name), cv.IMREAD_COLOR)
    hsv: np.ndarray = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    a = 10 < hsv[:, :, 0]
    b = hsv[:, :, 0] < 340
    c = hsv[:, :, 1] > 100
    d = hsv[:, :, 2] > 100
    result = (1 - a * b) * c * d
    result = result.astype(np.uint8)
    print(hsv.shape, result.shape)
    print(hsv.dtype, result.dtype)
    print(hsv[:, :, 2].max(), result.max())
    save(result, red_segments_dir / file.name)


def black_segmentation(file):
    image: np.ndarray = cv.imread(str(original_dir / file.name), cv.IMREAD_GRAYSCALE)
    result = image < 10
    result = result.astype(np.uint8)
    save(result, red_segments_dir / ('x' + file.name))


def find_best_k_and_size(get_distance, algorithm, grayscale_list, expected_list, k_values, size_values):
    min_error = None
    best_k = None
    best_window_size = None
    for window_size in size_values:
        for k in k_values:
            error = 0.0
            for grayscale, expected in zip(grayscale_list, expected_list):
                computed = algorithm(grayscale, k=k, window_size=window_size)
                error += get_distance(computed, expected)
            if min_error is None or min_error > error:
                min_error = error
                best_k = k
                best_window_size = window_size
    return best_k, best_window_size


def print_best_params():
    expected_list = []
    grayscale_list = []

    for file in list(sorted(expected_dir.iterdir())):
        expected_list.append(cv.imread(str(file), cv.IMREAD_GRAYSCALE) // 255)
        grayscale_list.append(cv.imread(str(grayscale_dir / file.name), cv.IMREAD_GRAYSCALE))

    def invert(f):
        def wrap(a, b):
            return -f(a, b)

        return wrap

    def thresh_to_image(thresh):
        def wrap(image: np.ndarray, **kwargs):
            return (image > thresh(image, **kwargs)).astype(np.uint8)

        return wrap

    distances = {'ssim': invert(ssim), 'mse': mse, 'cpm_norm': cpm, 'cpm': cpm_simple}
    algorithms = {'niblack': thresh_to_image(threshold_niblack), 'sauvola': thresh_to_image(threshold_sauvola)}

    k_values = list(np.arange(-0.8, 0.8, 0.2))
    size_values = range(15, 200, 30)
    for algorithm_name, algorithm_func in algorithms.items():
        for distance_name, distance_func in distances.items():
            if distance_name not in {'ssim', 'mse'} and False:
                # niblack: k=0.5999999999999996, window_size=195 with metric ssim
                # niblack: k=0.5999999999999996, window_size=195 with metric mse
                # niblack: k=0.5999999999999996, window_size=195 with metric psnr
                # niblack: k=0.5999999999999996, window_size=195 with metric cpm_norm
                # niblack: k=0.5999999999999996, window_size=165 with metric cpm
                # sauvola: k=0.3999999999999997, window_size=75 with metric ssim
                # sauvola: k=0.5999999999999996, window_size=45 with metric mse
                # sauvola: k=0.5999999999999996, window_size=45 with metric psnr
                # sauvola: k=0.19999999999999973, window_size=105 with metric cpm_norm
                # sauvola: k=0.19999999999999973, window_size=195 with metric cpm
                continue
            k, size = find_best_k_and_size(
                distance_func, algorithm_func, grayscale_list, expected_list, k_values, size_values)
            print(f'{algorithm_name}: k={k}, window_size={size} with metric {distance_name}')


convert_to_grayscale()


def get_k(h, w):
    x = np.tile(np.arange(w), (h, 1))
    y = np.tile(np.arange(h), (w, 1)).T
    print(x.dtype, y.dtype)
    kx = np.maximum(x, w - x) / w
    ky = np.maximum(y, h - y) / h
    return np.maximum(kx, ky)


if __name__ == '__main__':
    # output_dir = Path(__file__).parent.joinpath('images/output')
    # for path in grayscale_dir.iterdir():
    #     grayscale = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    # otsu = grayscale > threshold_otsu(grayscale)
    # save(otsu, output_dir / 'otsu' / path.name)
    # sauvola = grayscale > threshold_sauvola(grayscale, window_size=15, k=0.2)
    # save(sauvola, output_dir / 'sauvola' / path.name)
    # path = original_dir / 'image00003.jpg'
    # grayscale = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    # kwargs = dict(k=0.19999999999999973, window_size=105)
    # computed = (grayscale > threshold_sauvola(grayscale, **kwargs)).astype(np.uint8)
    # plt.imshow(computed, cmap='gray')
    # plt.show()
    # cv.threshold()
    # print_best_params()
    # convert_and_save_image(grayscale_dir / '0image_with_ink.png')
    # black_segmentation(original_dir / 'image00003.jpg')
    # red_segmentation(original_dir / '0image_with_ink.png')
    # path = original_dir / 'image00003.jpg'
    # path = original_dir / '3Sobornoe_Ulozhenie.jpg'
    path = original_dir / '1first-image.png'
    image: np.ndarray = cv.imread(str(path), cv.IMREAD_COLOR)
    red = image[:, :, 2]
    hsv: np.ndarray = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    src = val
    src = cv.morphologyEx(src, cv.MORPH_OPEN, np.ones((7, 7)))
    # src = cv.GaussianBlur(val, (3, 3), 0)
    # laplac = cv.Laplacian(src, cv.CV_16S)
    # edges_image = np.where(laplac > np.quantile(laplac, 0.9), 255, 0)
    # hist = np.histogram(val, weights=laplac)
    # val_new = val > threshold_otsu(val)
    # mask = np.where((1 - (15 <= hue) * (hue <= 170)) * (sat > 100) * (val > 100), 255, 0).astype(np.uint8)
    # only_dark = np.where(val < 100, 255, 0).astype(np.uint8)
    # hue_without_background = [x for x in hue.flatten() if x >= 25 or x <= 20]
    # mask = np.where(red < np.quantile(red, 0.05), 255, 0)
    edges_image = cv.Canny(src, 50, 200, None, 3)
    edges_image = (edges_image * get_k(edges_image.shape[0], edges_image.shape[1])).astype(np.uint8)
    lines = cv.HoughLines(edges_image, 1, np.pi / 180, 150, None, 0, 0)
    # plt.hist(red.flatten(), bins=255, density=True)
    # plt.hist(val.flatten(), bins=255, density=True)
    # plt.hist(sat.flatten(), bins=255, density=True)
    # plt.hist(hue.flatten(), bins=179, density=True)
    # plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    # plt.imshow(edges_image, cmap='gray', vmin=0, vmax=255)
    hough = np.zeros_like(edges_image)
    print(edges_image.shape)
    if lines is not None:
        print(len(lines))
        for line in lines[:10]:
            rho, theta = line[0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            print(pt1, pt2)
            cv.line(edges_image, pt1, pt2, 255, 3, cv.LINE_AA)
    plt.imshow(edges_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
