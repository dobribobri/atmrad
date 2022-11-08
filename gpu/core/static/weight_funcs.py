#  -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import Tensor1D_or_3D
from cpu.core.const import *
from gpu.core.static.water import dielectric
import gpu.core.math as math


def kw(frequency: float, t: Union[float, Tensor1D_or_3D], mode='840-8') -> Union[float, Tensor1D_or_3D]:
    """
    :param frequency: частота излучения в ГГц
    :param t: профиль температуры облаков или средняя эффективная температура облаков, град. Цельс.
    :param mode: модель расчета диэлектрической проницаемости
    :return: весовая функция k_w (вода в жидкокапельной фазе).
    """
    if mode in [1, 'one-dimensional']:
        lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
        eps = dielectric.epsilon_complex(frequency, t, mode='one-dimensional')
        return -0.6 * PI / lamda * math.im((eps - 1) / (eps + 2))

    # if mode in ['one-dimensional-wrong']:
    #     lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
    #     eps = dielectric.epsilon_complex(frequency, t, mode='one-dimensional')
    #     re = math.re(eps)
    #     im = -math.im(eps)
    #     eps = math.complex_(re, im)
    #     return 0.6 * PI / lamda * math.im((eps - 1) / (eps + 2))

    if mode in [2, 'one-dimensional-simplified']:
        lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
        epsO, epsS, lambdaS = dielectric.epsilon(t, 0.)
        y = lambdaS / lamda
        return 3 * 0.6 * PI / lamda * (epsS - epsO) * y / (
                (epsS + 2) * (epsS + 2) + (epsO + 2) * (epsO + 2) * y * y)

    if mode in [3, 'two-dimensional-c']:
        lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
        eps = dielectric.epsilon_complex(frequency, t, mode='two-dimensional')
        return -0.6 * PI / lamda * math.im((eps - 1) / (eps + 2))

    # if mode in ['two-dimensional-c-wrong']:
    #     lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
    #     eps = dielectric.epsilon_complex(frequency, t, mode='two-dimensional')
    #     re = math.re(eps)
    #     im = -math.im(eps)
    #     eps = math.complex_(re, im)
    #     return 0.6 * PI / lamda * math.im((eps - 1) / (eps + 2))

    if mode in [4, 'two-dimensional-b']:
        f = math.as_tensor(frequency)
        eps = dielectric.epsilon_complex(f, t, mode='two-dimensional')
        re = math.re(eps)
        im = -math.im(eps)
        eta = (2 + re) / im
        return 0.819 * f / (im * (1 + eta * eta)) * dB2np

    # Rec. ITU-R 840-8
    f = math.as_tensor(frequency)
    eps = dielectric.epsilon_complex(f, t, mode='two-dimensional')
    re = math.re(eps)
    im = -math.im(eps)
    eta = (2 + re) / im
    return 0.819 * (1.9479 / 10000 * math.pow_(f, 2.308) +
                    2.9424 * math.pow_(f, 0.7436) - 4.9451) / (im * (1 + eta * eta)) * dB2np
