#  -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import TensorLike
import gpu.core.math as math


"""
Водяной пар
"""


def pressure(T: Union[float, TensorLike], rho: Union[float, TensorLike]) -> Union[float, TensorLike]:
    """
    Парциальное давление водяного пара

    :param rho: плотность водяного пара (абсолютная влажность), г/м^3
    :param T: температура воздуха, град. Цельс.
    :return: давление в гПа
    """
    return rho * (T + 273.15) / 216.7


def relative_humidity(T: Union[float, TensorLike], P: Union[float, TensorLike],
                      rho: Union[float, TensorLike], method='wmo2008') -> Union[float, TensorLike]:
    """
    Расчет относительной влажности по абсолютной

    :param T: температура воздуха, град. Цельс.
    :param P: барометрическое давление, гПа
    :param rho: абсолютная влажность, г/м^3
    :param method: метод расчета давления насыщенного водяного пара
        ('wmo2008', 'august-roche-magnus', 'tetens', 'august', 'buck')
    :return: %
    """
    return pressure(T, rho) / saturated.pressure(T, P, method) * 100


def absolute_humidity(T: Union[float, TensorLike], P: Union[float, TensorLike],
                      rel: Union[float, TensorLike], method='wmo2008') -> Union[float, TensorLike]:
    """
    Расчет абсолютной влажности по относительной

    :param T: температура воздуха, град. Цельс.
    :param P: барометрическое давление, гПа
    :param rel: относительная влажность, %
    :param method: метод расчета давления насыщенного водяного пара
        ('wmo2008', 'august-roche-magnus', 'tetens', 'august', 'buck')
    :return: г/м^3
    """
    return (rel / 100) * 216.7 * saturated.pressure(T, P, method) / (T + 273.15)


class saturated:
    """
    Насыщенный водяной пар
    """

    @staticmethod
    def pressure(T: Union[float, TensorLike], P: Union[float, TensorLike] = None,
                 method='wmo2008') -> Union[float, TensorLike]:
        """
        Давление насыщенного водяного пара во влажном воздухе

        :param T: температура воздуха, град. Цельс.
        :param P: барометрическое давление, гПа
        :param method: метод аппроксимации ('wmo2008', 'august-roche-magnus',
            'tetens', 'august', 'buck')
        :return: давление в гПа
        """
        if method.lower() == 'august-roche-magnus':
            e = 0.61094 * math.exp(17.625 * T / (243.04 + T)) * 10
        elif method.lower() == 'tetens':
            e = 0.61078 * math.exp(17.27 * T / (T + 237.3)) * 10
        elif method.lower() == 'august':
            e = math.exp(20.386 - 5132 / (T + 273.15)) * 1.333
        elif method.lower() == 'buck':
            if T > 0:
                e = 6.1121 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
            else:
                e = 6.1115 * math.exp((23.036 - T / 333.7) * (T / (279.82 + T)))
        else:
            e = 6.112 * math.exp(17.62 * T / (243.12 + T))
        if P is None:
            return e
        return (1.0016 + 3.15 * 0.000001 * P - 0.074 / P) * e
