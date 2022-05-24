import numpy as np
from dataclasses import dataclass

from features import vertical_projection, horizontal_projection
from gradient import get_partial_y, get_partial_x


@dataclass
class Box:
    top: int
    right: int
    bottom: int
    left: int


def get_borders(projection, threshold=0):
    borders = []
    i = 0
    while i < len(projection):
        current = projection[i]
        if current > threshold:
            left = i
            while i + 1 < len(projection) and projection[i] > threshold:
                i += 1
            right = i
            borders.append((left, right))
        i += 1
    return borders


def get_symbol_boxes(binary: np.ndarray, threshold=0):
    projection = horizontal_projection(binary)
    borders = get_borders(projection)
    rects = []
    for left, right in borders:
        region = binary[:, left:right]
        diff = list(vertical_projection(region)[::-1])
        top = 0
        bottom = len(diff)
        for num, value in enumerate(diff):
            if value >= threshold and top == 0:
                top = num
                break
        for num, value in enumerate(diff[top+1:], top+1):
            if value >= threshold:
                bottom = num
        box = Box(top=top, right=right, bottom=bottom, left=left)
        rects.append(box)
    return rects
