import numpy as np
from matplotlib import pyplot as plt


def _reverse(binary: np.ndarray) -> np.ndarray:
    return 1 - binary


def _moment(binary: np.ndarray, x_exp: int = 0, y_exp: int = 0, x_offset: float = 0, y_offset: float = 0) -> float:
    h, w = binary.shape
    arr = _reverse(binary)
    if x_exp > 0:
        x = np.tile(np.arange(0, w), (h, 1))
        arr *= (x - x_offset) ** x_exp
    if y_exp > 0:
        y = np.tile(np.arange(0, h), (w, 1)).T
        arr *= (y - y_offset) ** y_exp
    return arr.sum(dtype=np.float)


def raw_moment(binary: np.ndarray, x_exp: int = 0, y_exp: int = 0):
    return _moment(binary, x_exp=x_exp, y_exp=y_exp)


def center(binary: np.ndarray) -> tuple[float, float]:
    w = _moment(binary)
    return _moment(binary, x_exp=1) / w, _moment(binary, y_exp=1) / w


def central_moment(binary: np.ndarray, x_exp: int = 0, y_exp: int = 0):
    x, y = center(binary)
    return _moment(binary, x_exp=x_exp, y_exp=y_exp, x_offset=x, y_offset=y)


def normalize_binary(binary: np.ndarray, zeros=False):
    """Расширяет чёрно-белую картинку пустыми полями

    :param binary: чёрно-белая картинка
    :param bool zeros: заполнять ли поля нулями вместо единиц
    :return:
    """
    x, y = center(binary)
    h, w = binary.shape
    y_shift = 2 * y - (h - 1)
    x_shift = 2 * x - (w - 1)
    new_shape = h + int(abs(y_shift)), w + int(abs(x_shift))
    if zeros:
        new_binary = np.zeros(new_shape)
    else:
        new_binary = np.ones(new_shape)
    left, right = (0, w) if x_shift > 0 else (-w, None)
    top, bottom = (0, h) if y_shift > 0 else (-h, None)
    new_binary[top:bottom, left:right] = binary
    return new_binary


def scale_invariant(binary: np.ndarray, x_exp: int = 0, y_exp: int = 0):
    normalized_binary = normalize_binary(binary)
    zero_binary = np.zeros_like(normalized_binary)
    return central_moment(normalized_binary, x_exp=x_exp, y_exp=y_exp) / central_moment(zero_binary, x_exp=x_exp, y_exp=y_exp)


def weight(binary: np.ndarray) -> float:
    return _moment(binary)


def density(binary: np.ndarray) -> float:
    h, w = binary.shape
    return weight(binary) / (w * h)


def relative_center(binary: np.ndarray) -> tuple[float, float]:
    h, w = binary.shape
    x, y = center(binary)
    return x / w, y / h


def axial_moments(binary: np.ndarray) -> tuple[float, float]:
    return central_moment(binary, y_exp=2), central_moment(binary, x_exp=2)


def relative_axial_moments(binary: np.ndarray) -> tuple[float, float]:
    return scale_invariant(binary, y_exp=2), scale_invariant(binary, x_exp=2)


def cov(binary: np.ndarray) -> np.ndarray:
    a = central_moment(binary, x_exp=2) / central_moment(binary)
    b = central_moment(binary, x_exp=1, y_exp=1) / central_moment(binary)
    c = central_moment(binary, y_exp=2) / central_moment(binary)
    return np.array([[a, b], [b, c]])


def orientation_angle(binary: np.ndarray) -> float:
    matrix = cov(binary)
    return np.arctan(2 * matrix[0, 1] / (matrix[0, 0] - matrix[1, 1])) / 2


def horizontal_projection(binary: np.ndarray, zero_weight=False) -> np.ndarray:
    if not zero_weight:
        binary = _reverse(binary)
    return binary.sum(axis=0).astype(np.uint8)


def vertical_projection(binary: np.ndarray, zero_weight=False) -> np.ndarray:
    if not zero_weight:
        binary = _reverse(binary)
    return binary.sum(axis=1).astype(np.uint8)[::-1]


def draw_projection(x, y, path, show: bool = False):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(x, y)
    axes.set_ylim(bottom=0)
    axes.set_xlim(left=0)
    if show:
        fig.show()
    fig.savefig(str(path), bbox_inches='tight')
    plt.close(fig)


from PIL import Image
from resampling import interpolation

if __name__ == '__main__':
    black = Image.open('black.bmp').convert('L')
    white = Image.open('white.bmp').convert('L')

    np_black = np.asarray(black) / 255
    np_black2 = normalize_binary(np_black)
    print(np_black.shape, np_black2.shape)
    assert (np_black - np_black2).sum() == 0

    np_white = np.asarray(white) / 255
    np_white2 = normalize_binary(np_white)
    print(np_white.shape, np_white2.shape)
    assert (np_white - np_white2).sum() == 0

    print('black:', relative_axial_moments(np_black))
    # сравнение относительных моментов для изображения чёрной рамки с различными интерполяциями
    print('original:', relative_axial_moments(np_white))
    print('x2:', relative_axial_moments(np.asarray(interpolation(white, 2)) / 255))
    print('x3:', relative_axial_moments(np.asarray(interpolation(white, 3)) / 255))
    print('x4:', relative_axial_moments(np.asarray(interpolation(white, 4)) / 255))
    print('x5:', relative_axial_moments(np.asarray(interpolation(white, 5)) / 255))
    print('x6:', relative_axial_moments(np.asarray(interpolation(white, 6)) / 255))
    print('x7:', relative_axial_moments(np.asarray(interpolation(white, 7)) / 255))
