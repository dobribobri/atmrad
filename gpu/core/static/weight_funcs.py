#  -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import Tensor1D_or_3D
from cpu.core.const import *
from gpu.core.static.water import dielectric
import gpu.core.math as math


def kw(frequency: float, t: Union[float, Tensor1D_or_3D]) -> Union[float, Tensor1D_or_3D]:
    """
    :param frequency: частота излучения в ГГц
    :param t: профиль температуры облаков или средняя эффективная температура облаков, град. Цельс.
    :return: весовая функция k_w (вода в жидкокапельной фазе).
    """
    # Б.Г. Кутуза (2)
    # lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
    # epsO, epsS, lambdaS = dielectric.epsilon(t, 0.)
    # y = lambdaS / lamda
    # return 3 * 0.6 * PI / lamda * (epsS - epsO) * y / (
    #         (epsS + 2) * (epsS + 2) + (epsO + 2) * (epsO + 2) * y * y)

    # Б.Г. Кутуза (1)
    # eps = dielectric.epsilon_complex(frequency, t)
    # return 0.6 * PI / lamda * math.im((eps - 1) / (eps + 2))

    # Rec. ITU-R 840
    f = math.as_tensor(frequency)
    eps = dielectric.epsilon_complex(frequency, t, mode=2)
    re = math.re(eps)
    im = math.im(eps)
    eta = (2 + re) / im
    # return 0.819 * f / (im * (1 + eta * eta)) * dB2np
    return 0.819 * (1.9479 / 10000 * math.pow_(f, 2.308) +
                    2.9424 * math.pow_(f, 0.7436) - 4.9451) / (im * (1 + eta * eta)) * dB2np
