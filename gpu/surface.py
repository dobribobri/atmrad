# -*- coding: utf-8 -*-
from typing import Union
from gpu.core.types import Tensor2D
import gpu.core.math as math
import gpu.core.static.water.Fresnel as Fresnel
import numpy as np


class Surface:
    def __init__(self, temperature: Union[float, Tensor2D],
                 theta: float = 0.,
                 polarization: str = None,
                 **kwargs):
        self._T = math.as_tensor(temperature)
        self._theta = theta
        self._polarization = polarization

        for name, value in kwargs.items():
            self.__setattr__(name, value)

    @property
    def temperature(self) -> Union[float, Tensor2D]:
        return self._T

    @temperature.setter
    def temperature(self, val: Union[float, Tensor2D]):
        self._T = math.as_tensor(val)

    @property
    def angle(self) -> float:
        return self._theta

    @angle.setter
    def angle(self, val: float):
        self._theta = val

    @property
    def polarization(self) -> str:
        return self._polarization

    @polarization.setter
    def polarization(self, val: str):
        self._polarization = val

    def reflectivity(self, frequency: float) -> Union[float, Tensor2D]:
        pass

    def emissivity(self, frequency: float) -> Union[float, Tensor2D]:
        pass


class SmoothWaterSurface(Surface):
    """
    Модель микроволнового излучения гладкой водной поверхности
    """
    def __init__(self, temperature: Union[float, Tensor2D] = 15.,
                 salinity: Union[float, Tensor2D] = 0.,
                 theta: float = 0., polarization: str = None):
        """
        :param temperature: термодинамическая температура поверхности, град. Цельс.
        :param salinity: соленость, промили
        :param theta: зенитный угол, рад.
        :param polarization: поляризация ('H' или 'V')
        """
        super().__init__(temperature, theta, polarization)
        self._Sw = math.as_tensor(salinity)

    @property
    def salinity(self) -> Union[float, Tensor2D]:
        return self._Sw

    @salinity.setter
    def salinity(self, val: Union[float, Tensor2D]):
        self._Sw = math.as_tensor(val)

    def reflectivity(self, frequency: float) -> Union[float, Tensor2D]:
        """
        Расчет отражательной способности

        :param frequency: частота излучения в ГГц
        :return: коэффициент отражения гладкой водной поверхности
        """
        if np.isclose(self._theta, 0.):
            ret = Fresnel.R(frequency, self._T, self._Sw)
        elif self._polarization in ['H', 'h']:
            ret = Fresnel.R_horizontal(frequency, self._theta, self._T, self._Sw)
        else:
            ret = Fresnel.R_vertical(frequency, self._theta, self._T, self._Sw)
        return math.as_tensor(ret)

    def emissivity(self, frequency: float) -> Union[float, Tensor2D]:
        """
        Расчет излучательной способности

        :param frequency: частота излучения в ГГц
        :return: коэффициент излучения гладкой водной поверхности при условии термодинамического равновесия
        """
        return 1. - self.reflectivity(frequency)
