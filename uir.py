import numpy as np

from features import weight


def cpm(real, expected):
    """count pseudometric measure - сравнение результатов бинаризации на основе количества чёрного

    источник: http://www.isa.ru/proceedings/images/documents/2013-63-3/t-3-13_85-94.pdf
    """
    if not isinstance(real, np.ndarray):
        real = np.asarray(real) / 255
    if not isinstance(expected, np.ndarray):
        expected = np.asarray(expected) / 255
    print(f'in cpm: weight(real) == {weight(real)}, weight(expected) == {weight(expected)}')
    return abs(weight(real) - weight(expected))
