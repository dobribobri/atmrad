#  -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import TensorLike
from cpu.core.const import *
import gpu.core.static.p676 as p676
import gpu.core.static.weight_funcs as wf


"""
Погонные коэффициенты поглощения (ослабления)
"""


def oxygen(frequency: float,
           T: Union[float, TensorLike], P: Union[float, TensorLike],
           rho: Union[float, TensorLike] = None, approx: bool = False) -> Union[float, TensorLike]:
    """
    Копия static.p676.gamma_oxygen(...)

    :param frequency: частота излучения в ГГц
    :param T: термодинамическая температура, градусы Цельсия
    :param P: атмосферное давление, мбар или гПа
    :param rho: абсолютная влажность, г/м^3
    :param approx: расчет по приближенной формуле
    :return: погонный коэффициент поглощения в кислороде (Дб/км)
    """
    if approx:
        return p676.gamma_oxygen_approx(frequency, T, P)
    return p676.gamma_oxygen(frequency, T, P, rho)


def water_vapor(frequency: float,
                T: Union[float, TensorLike], P: Union[float, TensorLike],
                rho: Union[float, TensorLike], approx: bool = False) -> Union[float, TensorLike]:
    """
    Копия static.p676.gamma_water_vapor(...)

    :param frequency: частота излучения в ГГц
    :param T: термодинамическая температура, градусы Цельсия
    :param P: атмосферное давление, мбар или гПа
    :param rho: абсолютная влажность, г/м^3
    :param approx: расчет по приближенной формуле
    :return: погонный коэффициент поглощения в водяном паре (Дб/км)
    """
    if approx:
        return p676.gamma_water_vapor_approx(frequency, T, P, rho)
    return p676.gamma_water_vapor(frequency, T, P, rho)


def liquid_water_eff(frequency: float,
                     t_clouds: float, w: Union[float, TensorLike]) -> Union[float, TensorLike]:
    """
    Б.Г. Кутуза

    :param frequency: частота излучения в ГГц
    :param t_clouds: средняя эффективная температура облаков, град. Цельс.
    :param w: поле водности, кг/м^3
    :return: погонный коэффициент поглощения в облаке (Дб/км)
    """
    return np2dB * wf.kw(frequency, t_clouds) * w


def liquid_water(frequency: float,
                 T: Union[float, TensorLike], w: Union[float, TensorLike]) -> Union[float, TensorLike]:
    """
    Б.Г. Кутуза

    :param frequency: частота излучения в ГГц
    :param T: профиль температуры облаков или средняя эффективная температура облаков, град. Цельс.
    :param w: поле водности, кг/м^3
    :return: погонный коэффициент поглощения в облаке (Дб/км)
    """
    return np2dB * wf.kw(frequency, T) * w
