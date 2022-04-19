# -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import Tensor2D
import gpu.core.math as math
from gpu.atmosphere import Atmosphere
from gpu.surface import Surface

"""
Спутник
"""


def brightness_temperature(frequency: float, atm: Atmosphere,
                           srf: 'Surface', cosmic: bool = True) -> Union[float, Tensor2D]:
    """
    Яркостная температура уходящего излучения системы 'атмосфера - подстилающая поверхность'

    :param frequency: частота излучения в ГГц
    :param atm: объект Atmosphere (атмосфера)
    :param srf: объект Surface (поверхность)
    :param cosmic: учитывать реликтовый фон
    """
    tau_exp = math.exp(-1 * atm.opacity.summary(frequency))
    tb_down = atm.downward.brightness_temperature(frequency, background=cosmic)
    tb_up = atm.upward.brightness_temperature(frequency)
    r = srf.reflectivity(frequency)
    kappa = 1. - r  # emissivity
    return math.as_tensor(srf.temperature + 273.15) * kappa * tau_exp + tb_up + r * tb_down * tau_exp
