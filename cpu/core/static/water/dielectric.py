#  -*- coding: utf-8 -*-
from typing import Union, Tuple
from cpu.core.types import TensorLike
from cpu.core.const import *
import cpu.core.math as math

"""
Диэлектрическая проницаемость воды с учетом солености
"""


def epsilon(T: Union[float, TensorLike],
            Sw: Union[float, TensorLike] = 0.) -> Tuple[float, float, float]:
    """
    :param T: термодинамическая температура воды, град. Цельс.
    :param Sw: соленость, промили
    :return: кортеж значений: 1 - оптическая составляющая диэлектрической проницаемости,
        2 - статическая составляющая, 3 - характерная длина волны
    """
    epsO_nosalt = 5.5
    epsS_nosalt = 88.2 - 0.40885 * T + 0.00081 * T * T
    lambdaS_nosalt = 1.8735116 - 0.027296 * T + 0.000136 * T * T + 1.662 * math.exp(-0.0634 * T)
    epsO = epsO_nosalt
    epsS = epsS_nosalt - 17.2 * Sw / 60
    lambdaS = lambdaS_nosalt - 0.206 * Sw / 60
    return epsO, epsS, lambdaS


def epsilon_complex(frequency: float, T: Union[float, TensorLike],
                    Sw: Union[float, TensorLike] = 0., mode='Rec.ITU-R P.840-8') -> Union[complex, TensorLike]:
    """
    Комплексная диэлектрическая проницаемость воды

    :param frequency: частота излучения в ГГц
    :param T: термодинамическая температура воды, град. Цельс.
    :param Sw: соленость, промили
    :param mode: выбор модели
    """

    if mode in [0, 'one-dimensional']:   # One-dimensional Debye formula
        lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
        epsO, epsS, lambdaS = epsilon(T, Sw)
        y = lambdaS / lamda
        eps1 = epsO + (epsS - epsO) / (1 + y * y)
        eps2 = y * (epsS - epsO) / (1 + y * y)
        sigma = 0.00001 * (2.63 * T + 77.5) * Sw
        eps2 = eps2 + 60 * sigma * lamda
        return math.complex_(eps1, -eps2)

    # Two-dimensional
    # Rec. ITU-R 840    # 840-8 - in force, main
    f = frequency
    theta = 300 / (T + 273.15)
    eps0 = 77.6 + 103.3 * (theta - 1)
    # eps1 = 5.48
    eps1 = 0.0671 * eps0
    # eps2 = 3.51
    eps2 = 3.52
    # fp = 20.09 - 142 * (theta - 1) + 294 * (theta - 1) * (theta - 1)
    fp = 20.09 - 146 * (theta - 1) + 316 * (theta - 1) * (theta - 1)
    # fs = 590 - 1500 * (theta - 1)
    fs = 39.8 * fp
    im = f * (eps0 - eps1) / (fp * (1 + (f / fp) * (f / fp))) + \
        f * (eps1 - eps2) / (fs * (1 + (f / fs) * (f / fs)))
    re = (eps0 - eps1) / (1 + (f / fp) * (f / fp)) + \
         (eps1 - eps2) / (1 + (f / fs) * (f / fs)) + eps2
    return math.complex_(re, -im)

