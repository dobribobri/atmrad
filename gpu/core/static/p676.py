#  -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import TensorLike, Tensor2D
from cpu.core.const import *
import gpu.core.math as math

"""
Рекомендация Международного Союза Электросвязи Rec.ITU-R P.676-3
"""


def H1(frequency: float) -> float:
    """
    :param frequency: частота излучения в ГГц
    :return: характеристическая высота поглощения в кислороде (км)
    """
    f = math.as_tensor(frequency)
    const = 6.
    if f < 50:
        return const
    elif 70 < f < 350:
        return const + 40 / ((f - 118.7) * (f - 118.7) + 1)
    return const


def H2(frequency: float, rainQ: bool = False) -> float:
    """
    :param frequency: частота излучения в ГГц
    :param rainQ: идет дождь? True/False
    :return: характеристическая высота поглощения в водяном паре (км)
    """
    f = math.as_tensor(frequency)
    Hw = 1.6
    if rainQ:
        Hw = 2.1
    return Hw * (1 + 3. / ((f - 22.2) * (f - 22.2) + 5) + 5. / ((f - 183.3) * (f - 183.3) + 6) +
                 2.5 / ((f - 325.4) * (f - 325.4) + 4))


def gamma_oxygen(frequency: float,
                 T: Union[float, TensorLike], P: Union[float, TensorLike]) -> Union[float, TensorLike]:
    """
    :param frequency: частота излучения в ГГц
    :param T: термодинамическая температура, градусы Цельсия
    :param P: атмосферное давление, мбар или гПа
    :return: погонный коэффициент поглощения в кислороде (Дб/км)
    """
    rp = P / 1013
    rt = 288 / (273 + T)
    f = math.as_tensor(frequency)
    gamma = 0
    if f <= 57:
        gamma = (7.27 * rt / (f * f + 0.351 * rp * rp * rt * rt) +
                 7.5 / ((f - 57) * (f - 57) + 2.44 * rp * rp * rt * rt * rt * rt * rt)) * \
                f * f * rp * rp * rt * rt / 1000
    elif 63 <= f <= 350:
        gamma = (2 / 10000 * math.pow_(rt, 1.5) * (1 - 1.2 / 100000 * math.pow_(f, 1.5)) +
                 4 / ((f - 63) * (f - 63) + 1.5 * rp * rp * rt * rt * rt * rt * rt) +
                 0.28 * rt * rt / ((f - 118.75) * (f - 118.75) + 2.84 * rp * rp * rt * rt)) * \
                f * f * rp * rp * rt * rt / 1000
    elif 57 < f < 63:
        gamma = (f - 60) * (f - 63) / 18 * gamma_oxygen(57., T, P) - \
                1.66 * rp * rp * math.pow_(rt, 8.5) * (f - 57) * (f - 63) + \
                (f - 57) * (f - 60) / 18 * gamma_oxygen(63., T, P)
    return gamma


def gamma_water_vapor(frequency: float,
                      T: Union[float, TensorLike], P: Union[float, TensorLike],
                      rho: Union[float, TensorLike]) -> Union[float, TensorLike]:
    """
    :param frequency: частота излучения в ГГц
    :param T: термодинамическая температура, градусы Цельсия
    :param P: атмосферное давление, мбар или гПа
    :param rho: абсолютная влажность, г/м^3
    :return: погонный коэффициент поглощения в водяном паре (Дб/км)
    """
    rp = P / 1013
    rt = 288 / (273 + T)
    f = math.as_tensor(frequency)
    gamma = 0
    if f <= 350:
        gamma = (3.27 / 100 * rt +
                 1.67 / 1000 * rho * rt * rt * rt * rt * rt * rt * rt / rp +
                 7.7 / 10000 * math.pow_(f, 0.5) +
                 3.79 / ((f - 22.235) * (f - 22.235) + 9.81 * rp * rp * rt) +
                 11.73 * rt / ((f - 183.31) * (f - 183.31) + 11.85 * rp * rp * rt) +
                 4.01 * rt / ((f - 325.153) * (f - 325.153) + 10.44 * rp * rp * rt)) * \
                f * f * rho * rp * rt / 10000
    return gamma


def tau_oxygen_near_ground(frequency: float,
                           T_near_ground: Union[float, Tensor2D],
                           P_near_ground: Union[float, Tensor2D],
                           theta: float = 0.0) -> Union[float, Tensor2D]:
    """
    Учитывает угол наблюдения.

    :param frequency: частота излучения в ГГц
    :param T_near_ground: значение или 2D-срез температуры приземного слоя воздуха, градусы Цельсия
    :param P_near_ground: значение или 2D-срез атмосферного давления, гПа
    :param theta: угол наблюдения в радианах
    :return: полное поглощение в кислороде (оптическая толщина). В неперах
    """
    gamma = gamma_oxygen(frequency, T_near_ground, P_near_ground)
    return gamma * H1(frequency) / math.cos(theta) * dB2np


def tau_water_vapor_near_ground(frequency: float,
                                T_near_ground: Union[float, Tensor2D],
                                P_near_ground: Union[float, Tensor2D],
                                rho_near_ground: Union[float, Tensor2D],
                                theta: float = 0.0, rainQ=False) -> Union[float, Tensor2D]:
    """
    Учитывает угол наблюдения.

    :param frequency: частота излучения в ГГц
    :param T_near_ground: значение или 2D-срез температуры приземного слоя воздуха, градусы Цельсия
    :param P_near_ground: значение или 2D-срез приповерхностного атмосферного давления, гПа
    :param rho_near_ground: значение или 2D-срез приповерхностной абсолютной влажности, г/м^3
    :param theta: угол наблюдения в радианах
    :param rainQ: идет дождь? True/False
    :return: полное поглощение в водяном паре. В неперах
    """
    gamma = gamma_water_vapor(frequency, T_near_ground, P_near_ground, rho_near_ground)
    return gamma * H2(frequency, rainQ=rainQ) / math.cos(theta) * dB2np
