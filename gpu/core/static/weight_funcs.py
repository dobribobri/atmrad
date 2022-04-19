#  -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import Tensor1D_or_3D
from cpu.core.const import *
from gpu.core.static.water import dielectric


def kw(frequency: float, t: Union[float, Tensor1D_or_3D]) -> Union[float, Tensor1D_or_3D]:
    """
    :param frequency: частота излучения в ГГц
    :param t: профиль температуры облаков или средняя эффективная температура облаков, град. Цельс.
    :return: весовая функция k_w (вода в жидкокапельной фазе).
    """
    lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
    epsO, epsS, lambdaS = dielectric.epsilon(t, 0.)
    y = lambdaS / lamda
    return 3 * 0.6 * PI / lamda * (epsS - epsO) * y / (
            (epsS + 2) * (epsS + 2) + (epsO + 2) * (epsO + 2) * y * y)
