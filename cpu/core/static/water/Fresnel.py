#  -*- coding: utf-8 -*-
from typing import Union
from cpu.core.types import Number, Tensor2D
from cpu.core.const import *
import cpu.core.math as math
import cpu.core.static.water.dielectric as dielectric


"""
Формулы Френеля
"""


def M_horizontal(frequency: float, psi: float, T: Union[float, Tensor2D],
                 Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
    """

    :param frequency: частота излучения в ГГц
    :param psi: угол скольжения, рад.
    :param T: температура поверхности, град. Цельс.
    :param Sw: соленость, промили
    """
    epsilon = dielectric.epsilon_complex(frequency, T, Sw)
    cos = math.sqrt(epsilon - math.complex_(math.cos(psi) * math.cos(psi)))
    return (math.complex_(math.sin(psi)) - cos) / (math.complex_(math.sin(psi)) + cos)


def M_vertical(frequency: float, psi: float, T: Union[float, Tensor2D],
               Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
    epsilon = dielectric.epsilon_complex(frequency, T, Sw)
    cos = math.sqrt(epsilon - math.complex_(real=math.cos(psi) * math.cos(psi)))
    return (epsilon * math.complex_(math.sin(psi)) - cos) / \
           (epsilon * math.complex_(math.sin(psi)) + cos)


def R_horizontal(frequency: float, theta: float, T: Union[float, Tensor2D],
                 Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
    """
    :param frequency: частота излучения в ГГц
    :param theta: зенитный угол, рад.
    :param T: температура поверхности, град. Цельс.
    :param Sw: соленость, промили
    :return: коэффициент отражения на горизонтальной поляризации
    """
    M_h = M_horizontal(frequency, PI / 2. - theta, T, Sw)
    val = math.abs_(M_h)
    return val * val


def R_vertical(frequency: float, theta: float, T: Union[float, Tensor2D],
               Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
    """
    :param frequency: частота излучения в ГГц
    :param theta: зенитный угол, рад.
    :param T: температура поверхности, град. Цельс.
    :param Sw: соленость, промили
    :return: коэффициент отражения на вертикальной поляризации
    """
    M_v = M_vertical(frequency, PI / 2. - theta, T, Sw)
    val = math.abs_(M_v)
    return val * val


def R(frequency: float, T: Union[float, Tensor2D],
      Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
    """
    :param frequency: частота излучения в ГГц
    :param T: температура поверхности, град. Цельс.
    :param Sw: соленость, промили
    :return: коэффициент отражения при зенитном угле 0 рад
    """
    epsilon = dielectric.epsilon_complex(frequency, T, Sw)
    val = math.abs_((math.sqrt(epsilon) - 1) / (math.sqrt(epsilon) + 1))
    return val * val
