# -*- coding: utf-8 -*-
from typing import Union, List
from cpu.core.types import Tensor2D
import cpu.core.math as math
from cpu.atmosphere import Atmosphere
from cpu.surface import Surface
from cpu.core.multi import parallel
import numpy as np

"""
Спутник
"""


def brightness_temperature(frequency: float,
                           atm: Atmosphere,
                           srf: 'Surface',
                           __theta: float = None,
                           cosmic: bool = True) -> Union[float, Tensor2D]:
    """
    Яркостная температура уходящего излучения системы 'атмосфера - подстилающая поверхность'

    :param frequency: частота излучения в ГГц
    :param atm: объект Atmosphere (атмосфера)
    :param srf: объект Surface (поверхность)
    :param __theta: угол наблюдения в радианах (deprecated)
    :param cosmic: учитывать реликтовый фон
    """
    tau_exp = math.exp(-1 * atm.opacity.summary(frequency, __theta))
    tb_down = atm.downward.brightness_temperature(frequency, __theta, background=cosmic)
    tb_up = atm.upward.brightness_temperature(frequency, __theta)
    if __theta:
        assert srf.angle == __theta, 'эти углы должны совпадать'
    r = srf.reflectivity(frequency)
    kappa = 1. - r  # emissivity
    return math.as_tensor(srf.temperature + 273.15) * kappa * tau_exp + tb_up + r * tb_down * tau_exp


def brightness_temperatures(frequencies: Union[np.ndarray, List[float]],
                            atm: 'Atmosphere',
                            srf: 'Surface',
                            __theta: float = None,
                            cosmic: bool = True,
                            n_workers: int = None) -> np.ndarray:
    """
    Яркостная температура уходящего излучения системы 'атмосфера - подстилающая поверхность'

    :param frequencies: список частот в ГГц
    :param atm: объект Atmosphere (атмосфера)
    :param srf: объект Surface (поверхность)
    :param __theta: угол наблюдения в радианах (deprecated)
    :param cosmic: учитывать реликтовый фон
    :param n_workers: количество потоков для распараллеливания
    """
    return parallel(frequencies, func=brightness_temperature, args=(atm, srf, __theta, cosmic, ), n_workers=n_workers)
