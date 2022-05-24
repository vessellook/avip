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
    return arr.sum()


def raw_moment(binary: np.ndarray, x_exp: int = 0, y_exp: int = 0):
    return _moment(binary, x_exp=x_exp, y_exp=y_exp)


def center(binary: np.ndarray) -> tuple[float, float]:
    w = _moment(binary)
    return _moment(binary, x_exp=1) / w, _moment(binary, y_exp=1) / w


def central_moment(binary: np.ndarray, x_exp: int = 0, y_exp: int = 0):
    x, y = center(binary)
    return _moment(binary, x_exp=x_exp, y_exp=y_exp, x_offset=x, y_offset=y)


def scale_invariant(binary: np.ndarray, x_exp: int = 0, y_exp: int = 0):
    return central_moment(binary, x_exp=x_exp, y_exp=y_exp) / (_moment(binary) ** (1 + (x_exp + y_exp) / 2))


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
