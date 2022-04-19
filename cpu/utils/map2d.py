#  -*- coding: utf-8 -*-

from typing import Union, Tuple
import numpy as np
from scipy.signal import convolve2d


def __(kernel: Union[Tuple, int]):
    if type(kernel) == int:
        ni = nj = kernel
    else:
        ni, nj = kernel
    return ni, nj


def block_averaging(array2d: np.ndarray, kernel: Union[Tuple, int] = (10, 10), same_size=True) -> np.ndarray:
    ni, nj = __(kernel)
    if same_size:
        for i in range(0, len(array2d), ni):
            for j in range(0, len(array2d[i]), nj):
                array2d[i:i + ni, j:j + nj] = np.mean(array2d[i:i + ni, j:j + nj])
    else:
        new_arr = np.zeros((len(array2d) // ni, len(array2d[0]) // nj), dtype=float)
        for i in range(0, len(array2d), ni):
            for j in range(0, len(array2d[i]), nj):
                new_arr[i // ni, j // nj] = np.mean(array2d[i:i + ni, j:j + nj])
        array2d = new_arr
    return array2d


def conv_averaging(array2d: np.ndarray, kernel: Union[Tuple, int] = (10, 10)) -> np.ndarray:
    ni, nj = __(kernel)
    kernel = np.ones((ni, nj)) / (ni * nj)
    return convolve2d(array2d, kernel, mode='valid').flatten()


def add_zeros(array2d: np.ndarray, bounds: Union[Tuple[int, int], Tuple[int, int, int, int], int]):
    if type(bounds) == int:
        top = right = bottom = left = bounds
    elif len(bounds) == 2:
        top, right = bounds
        bottom, left = top, right
    elif len(bounds) == 4:
        top, right, bottom, left = bounds
    else:
        raise RuntimeError('неверно задан параметр \'bounds\'')
    assert top >= 0 and right >= 0 and bottom >= 0 and left >= 0, 'только положительные числа'
    h, w = array2d.shape
    b = np.zeros((h + top + bottom, w + left + right), dtype=array2d.dtype)
    nh, nw = b.shape
    if bottom == 0:
        bottom = -nh
    if right == 0:
        right = -nw
    b[top:-bottom, left:-right] = array2d[:, :]
    return b
