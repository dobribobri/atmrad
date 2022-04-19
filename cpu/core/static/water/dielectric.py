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
                    Sw: Union[float, TensorLike] = 0.) -> Union[complex, TensorLike]:
    """
    Комплексная диэлектрическая проницаемость воды

    :param frequency: частота излучения в ГГц
    :param T: термодинамическая температура воды, град. Цельс.
    :param Sw: соленость, промили
    """
    lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
    epsO, epsS, lambdaS = epsilon(T, Sw)
    y = lambdaS / lamda
    eps1 = epsO + (epsS - epsO) / (1 + y * y)
    eps2 = y * (epsS - epsO) / (1 + y * y)
    sigma = 0.00001 * (2.63 * T + 77.5) * Sw
    eps2 = eps2 + 60 * sigma * lamda
    return math.complex_(eps1, -eps2)
