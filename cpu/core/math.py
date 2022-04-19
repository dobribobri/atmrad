#  -*- coding: utf-8 -*-
from typing import Union, List, Tuple
from cpu.core.types import Number, TensorLike, cpu_float
import numpy as np


def rank(a: Union[Number, TensorLike]) -> int:
    return np.ndim(a)


def sum_(a: TensorLike, axis: int = None) -> Union[Number, TensorLike]:
    return np.sum(a, axis=axis, dtype=cpu_float)


def transpose(a: TensorLike, axes=None) -> TensorLike:
    return np.transpose(a, axes)


def len_(a: TensorLike) -> int:
    return np.shape(a)[-1]


def exp(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return np.exp(a)


def log(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return np.log(a)


def sin(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return np.sin(a)


def cos(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return np.cos(a)


def tan(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return np.tan(a)


def sqrt(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return np.sqrt(a)


def abs_(a: Union[Number, TensorLike]) -> Union[float, TensorLike]:
    return np.absolute(a)


def pow_(a: Union[Number, TensorLike], d: float) -> Union[Number, TensorLike]:
    return np.power(a, d)


def as_tensor(a: Union[Number, TensorLike, List]) -> TensorLike:
    return np.asarray(a, dtype=cpu_float)


def as_variable(a: Union[Number, TensorLike]) -> TensorLike:
    return a


def zeros(shape: Union[int, List[int], Tuple[int]]) -> TensorLike:
    return np.zeros(shape, dtype=cpu_float)


def zeros_like(a: Union[Number, TensorLike]) -> TensorLike:
    return np.zeros_like(a, dtype=cpu_float)


def complex_(real: float, imag: float = 0.) -> complex:
    return np.complex(real, imag)


def round_(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return np.round(a)


def mean(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return np.mean(a, axis=axis, dtype=cpu_float)


def min_(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return np.min(a, axis=axis)


def max_(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return np.max(a, axis=axis)


def stddev(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return np.std(a, axis=axis, dtype=cpu_float)


def variance(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return np.var(a, axis=axis)
