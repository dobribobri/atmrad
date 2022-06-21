#  -*- coding: utf-8 -*-
from typing import Union, List, Tuple
from gpu.core.types import Number, TensorLike, gpu_float
import tensorflow as tf


def rank(a: Union[Number, TensorLike]) -> int:
    return tf.rank(a)


def shape(a: TensorLike):
    return tf.shape(a)


def sum_(a: TensorLike, axis: int = None) -> Union[Number, TensorLike]:
    return tf.reduce_sum(a, axis=axis)


def transpose(a: TensorLike, axes=None) -> TensorLike:
    return tf.transpose(a, perm=axes)


def len_(a: TensorLike) -> int:
    return tf.shape(a)[-1]


def exp(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return tf.exp(a)


def log(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return tf.math.log(a)


def sin(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return tf.sin(a)


def cos(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return tf.cos(a)


def tan(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return tf.tan(a)


def sqrt(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return tf.sqrt(a)


def abs_(a: Union[Number, TensorLike]) -> Union[float, TensorLike]:
    return tf.abs(a)


def pow_(a: Union[Number, TensorLike], d: float) -> Union[Number, TensorLike]:
    return tf.pow(a, d)


def as_tensor(a: Union[Number, TensorLike, List]) -> TensorLike:
    return tf.cast(tf.convert_to_tensor(a), dtype=gpu_float)


def as_variable(a: Union[Number, TensorLike]) -> TensorLike:
    return tf.Variable(a, dtype=gpu_float)


def zeros(shape: Union[int, List[int], Tuple[int]]) -> TensorLike:
    return tf.zeros(shape, dtype=gpu_float)


def zeros_like(a: Union[Number, TensorLike]) -> TensorLike:
    return tf.zeros_like(a, dtype=gpu_float)


def complex_(real: float, imag: float = 0.) -> complex:
    return tf.complex(as_tensor(real), as_tensor(imag))


def round_(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
    return tf.round(a)


def mean(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return tf.reduce_mean(a, axis=axis)


def min_(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return tf.reduce_min(a, axis=axis)


def max_(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return tf.reduce_max(a, axis=axis)


def stddev(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return tf.math.reduce_std(a, axis=axis)


def variance(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
    return tf.math.reduce_variance(a, axis=axis)


def move_axis(a: TensorLike, axis: int, destination: int) -> TensorLike:
    return tf.experimental.numpy.moveaxis(a, axis, destination)


def linalg_solve(a: TensorLike, b: TensorLike):
    return tf.linalg.solve(a, b)
