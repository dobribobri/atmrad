
# -*- coding: utf-8 -*-
from typing import Union, Tuple, List, Iterable, Callable
import numpy as np
import warnings
from cpu import TensorLike, Number
from cpu import op_cpu
from cpu import atmospheric
from cpu import ar as cpu

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

gpu_float = tf.float32


class op_gpu(op_cpu):
    @staticmethod
    def rank(a: TensorLike) -> int:
        return tf.rank(a)

    @staticmethod
    def sum(a: TensorLike, axis: int = None) -> Union[Number, TensorLike]:
        return tf.reduce_sum(a, axis=axis)

    @staticmethod
    def transpose(a: TensorLike, axes=None) -> TensorLike:
        return tf.transpose(a, perm=axes)

    @staticmethod
    def len(a: TensorLike) -> int:
        return tf.shape(a)[-1]

    @staticmethod
    def exp(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return tf.exp(a)

    @staticmethod
    def log(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return tf.math.log(a)

    @staticmethod
    def sin(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return tf.sin(a)

    @staticmethod
    def cos(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return tf.cos(a)

    @staticmethod
    def sqrt(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return tf.sqrt(a)

    @staticmethod
    def abs(a: Union[Number, TensorLike]) -> Union[float, TensorLike]:
        return tf.abs(a)

    @staticmethod
    def pow(a: Union[Number, TensorLike], d: float) -> Union[float, TensorLike]:
        return tf.pow(a, d)

    @staticmethod
    def as_tensor(a: Union[Number, TensorLike, Iterable[float]]) -> TensorLike:
        return tf.cast(tf.convert_to_tensor(a), dtype=gpu_float)

    @staticmethod
    def as_variable(a: Union[Number, TensorLike]) -> tf.Variable:
        return tf.Variable(a)

    @staticmethod
    def zeros(shape: Union[int, Iterable[int], Tuple[int]]) -> TensorLike:
        return tf.zeros(shape, dtype=gpu_float)

    @staticmethod
    def zeros_like(a: Union[Number, TensorLike, Iterable[float]]) -> TensorLike:
        return tf.zeros_like(a, dtype=gpu_float)

    @staticmethod
    def complex(real: float, imag: float) -> complex:
        return tf.complex(real, imag)

    @staticmethod
    def round(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return tf.round(a)

    @staticmethod
    def mean(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return tf.reduce_mean(a, axis=axis)

    @staticmethod
    def min(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return tf.reduce_min(a, axis=axis)

    @staticmethod
    def max(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return tf.reduce_max(a, axis=axis)

    @staticmethod
    def stddev(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return tf.math.reduce_std(a, axis=axis)

    @staticmethod
    def variance(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return tf.math.reduce_variance(a, axis=axis)


class ar(cpu):
    cpu.op = op_gpu

    class c(cpu.c):

        class multi:
            @staticmethod
            def parallel(frequencies: Union[np.ndarray, List[float]],
                         func: Callable, args: Union[Tuple, List, Iterable]) -> np.ndarray:
                return np.asarray([func(f, *args) for f in frequencies], dtype=object)

    class Atmosphere(cpu.Atmosphere):

        class downward(cpu.Atmosphere.downward):
            """
            Нисходящее излучение
            """
            @atmospheric
            def brightness_temperatures(self: 'ar.Atmosphere', frequencies: Union[np.ndarray, List[float]],
                                        n_workers: int = None) -> np.ndarray:
                """
                Яркостная температура нисходящего излучения

                :param frequencies: список частот в ГГц
                :param n_workers: количество потоков для распараллеливания
                    (в режиме gpu игнорируется)
                """
                if n_workers is not None:
                    warnings.warn('в gpu-режиме n_workers игнорируется')
                return ar.c.multi.parallel(frequencies,
                                           func=self.downward.brightness_temperature,
                                           args=())

        class upward(cpu.Atmosphere.upward):
            """
            Восходящее излучение
            """
            @atmospheric
            def brightness_temperatures(self: 'ar.Atmosphere', frequencies: Union[np.ndarray, List[float]],
                                        n_workers: int = None) -> np.ndarray:
                """
                Яркостная температура восходящего излучения (без учета подстилающей поверхности)

                :param frequencies: список частот в ГГц
                :param n_workers: количество потоков для распараллеливания
                    (в режиме gpu игнорируется)
                """
                if n_workers is not None:
                    warnings.warn('в gpu-режиме n_workers игнорируется')
                return ar.c.multi.parallel(frequencies,
                                           func=self.upward.brightness_temperature,
                                           args=())

    # noinspection PyArgumentList
    class satellite(cpu.satellite):
        """
        Спутник
        """
        class multi:
            @staticmethod
            def brightness_temperature(frequencies: Union[np.ndarray, List[float]], atm: 'ar.Atmosphere',
                                       srf: 'ar.Surface') -> np.ndarray:
                """
                Яркостная температура уходящего излучения системы 'атмосфера - подстилающая поверхность'

                :param frequencies: список частот в ГГц
                :param atm: объект Atmosphere (атмосфера)
                :param srf: объект Surface (поверхность)
                """
                return ar.c.multi.parallel(frequencies,
                                           func=ar.satellite.brightness_temperature,
                                           args=(atm, srf,))
