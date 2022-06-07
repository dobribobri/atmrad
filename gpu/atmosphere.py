#  -*- coding: utf-8 -*-
from typing import Tuple, Union
from functools import wraps
from gpu.core.types import Tensor1D_or_3D, Tensor2D, cpu_float
from cpu.core.const import *
import gpu.core.math as math
from gpu.core.common import at, cx
from gpu.core import attenuation
from gpu.core.static.water import vapor
import gpu.core.integrate as integrate
import numpy as np


def atmospheric(method):
    @wraps(method)
    def wrapper(obj: 'Atmosphere', *args, **kwargs):
        if hasattr(obj, 'outer'):
            obj = obj.outer
        return method(obj, *args, **kwargs)
    return wrapper


class Atmosphere:
    def __init__(self, Temperature: Tensor1D_or_3D, Pressure: Tensor1D_or_3D,
                 AbsoluteHumidity: Tensor1D_or_3D = None, RelativeHumidity: Tensor1D_or_3D = None,
                 LiquidWater: Tensor1D_or_3D = None, altitudes: np.ndarray = None, dh: float = None, **kwargs):
        """
        Модель собственного радиотеплового излучения атмосферы Земли

        :param Temperature: термодинамическая температура (высотный 1D-профиль или 3D-поле), град. Цельс.
        :param Pressure: атмосферное давление (1D или 3D), гПа
        :param AbsoluteHumidity: абсолютная влажность (1D или 3D), г/м^3. Параметр может быть не указан,
            если указан RelativeHumidity
        :param RelativeHumidity: относительная влажность (1D или 3D), %. Параметр может быть не указан,
            если указан AbsoluteHumidity
        :param LiquidWater: 1D-профиль или 3D-поле водности, кг/м^3. Параметр может быть не указан.
        :param altitudes: соответствующие высоты (1D массив), км. Может быть не указан, если указан параметр dh.
            Не может включать высоту h=0
        :param dh: постоянный шаг по высоте, км. Может быть не указан, если указаны altitudes
        """
        self._T = math.as_tensor(Temperature)
        del Temperature

        self._P = math.as_tensor(Pressure)
        del Pressure

        if AbsoluteHumidity is None:
            AbsoluteHumidity = vapor.absolute_humidity(self._T, self._P, RelativeHumidity)
            del RelativeHumidity
        self._rho = math.as_tensor(AbsoluteHumidity)
        del AbsoluteHumidity

        assert self._T.shape == self._P.shape == self._rho.shape, 'dimensions must match'

        if altitudes is None and dh is None:
            raise ValueError('please specify altitudes or dh')
        if altitudes is None:
            assert not np.isclose(dh, 0.), 'too small step dh'
            self._dh = np.cast[cpu_float](dh)  # self._dh - 1 number
            self._alt = np.cumsum([dh for _ in range(self._T.shape[-1])], dtype=cpu_float)   # self._alt - array
        else:
            assert self._T.shape[-1] == len(altitudes), 'lengths do not match'
            assert not np.isclose(altitudes[0], 0.), 'zero altitude not allowed'
            self._dh = np.diff(np.insert(altitudes, 0, 0.)).astype(cpu_float)  # self._dh - array
            self._alt = np.asarray(altitudes, dtype=cpu_float)  # self._alt - array
        del altitudes
        del dh

        if LiquidWater is None:
            LiquidWater = math.zeros_like(self._T)
        self._w = math.as_tensor(LiquidWater)   # распределение жидкокапельной влаги 1D или 3D
        del LiquidWater

        self._tcl = -2  # оценка на эффективную температуру облачности по Цельсию
        self._theta = 0.  # зенитный угол наблюдения в радианах
        self._PX = 50.  # горизонтальная протяженность в километрах
        self.incline = 'left'     # наклон траектории наблюдения (left/right) - учитывается, если theta != 0
        self.integration_method = 'boole'   # метод интегрирования
        self._use_tcl = False   # в расчетах использовать эффективную температуру облаков
        self.T_cosmic = 2.7    # температура реликтового фона в К

        for name, value in kwargs.items():
            self.__setattr__(name, value)

        self.attenuation = Atmosphere.attenuation(self)
        self.opacity = Atmosphere.opacity(self)
        self.downward = Atmosphere.downward(self)
        self.upward = Atmosphere.upward(self)

    @property
    def temperature(self) -> Tensor1D_or_3D:
        return self._T

    @temperature.setter
    def temperature(self, val: Tensor1D_or_3D):
        self._T = math.as_tensor(val)

    @property
    def pressure(self) -> Tensor1D_or_3D:
        return self._P

    @pressure.setter
    def pressure(self, val: Tensor1D_or_3D):
        self._P = math.as_tensor(val)

    @property
    def absolute_humidity(self) -> Tensor1D_or_3D:
        return self._rho

    @absolute_humidity.setter
    def absolute_humidity(self, val: Tensor1D_or_3D):
        self._rho = math.as_tensor(val)

    @property
    def relative_humidity(self) -> Tensor1D_or_3D:
        return vapor.relative_humidity(self._T, self._P, self._rho)

    @relative_humidity.setter
    def relative_humidity(self, val: Tensor1D_or_3D):
        self.absolute_humidity = math.as_tensor(vapor.absolute_humidity(self._T, self._P, val))

    @property
    def liquid_water(self) -> Tensor1D_or_3D:
        return self._w

    @liquid_water.setter
    def liquid_water(self, val: Tensor1D_or_3D):
        self._w = math.as_tensor(val)

    @property
    def altitudes(self) -> np.ndarray:
        return self._alt

    @altitudes.setter
    def altitudes(self, val: np.ndarray):
        assert not np.isclose(val[0], 0.), 'zero altitude not allowed'
        self._dh = np.diff(np.insert(val, 0, 0.)).astype(cpu_float)  # self._dh - array
        self._alt = np.asarray(val, dtype=cpu_float)  # self._alt - array

    @property
    def dh(self) -> Union[float, np.ndarray]:
        return self._dh

    @dh.setter
    def dh(self, val: float):
        # assert self._T.shape == self._P.shape == self._rho.shape, 'dimensions must match'
        self._alt = np.cumsum([val for _ in range(self._T.shape[-1])], dtype=cpu_float)  # self._alt - array
        self._dh = np.cast[cpu_float](val)  # self._dh - 1 number

    @property
    def effective_cloud_temperature(self) -> float:
        return self._tcl

    @effective_cloud_temperature.setter
    def effective_cloud_temperature(self, val: float):
        self._use_tcl = True
        self._tcl = val

    @property
    def angle(self) -> float:
        return self._theta

    @angle.setter
    def angle(self, val: float):
        self._theta = val

    @property
    def horizontal_extent(self):
        return self._PX

    @horizontal_extent.setter
    def horizontal_extent(self, val: float):
        self._PX = val

    @property
    def Q(self):
        return integrate.full(self._rho, self._dh, self.integration_method) / 10.

    @property
    def W(self):
        return integrate.full(self._w, self._dh, self.integration_method)

    @classmethod
    def Standard(cls, T0: float = 15., P0: float = 1013, rho0: float = 7.5,
                 altitudes: np.ndarray = None, H: float = 10, dh: float = 10. / 500,
                 beta: Tuple[float, float, float] = (6.5, 1., 2.8),
                 HP: float = 7.7, Hrho: float = 2.1) -> 'Atmosphere':
        """
        Стандартная атмосфера

        :param T0: приповерхностная температура, град. Цельс.
        :param P0: давление на уровне поверхности, гПа
        :param rho0: приповерхностное значение абсолютной влажности, г/м^3
        :param H: высота расчетной области, км
        :param dh: шаг по высоте, км
        :param altitudes: соответствующие высоты (1D массив), км. Может быть не указан, если указаны параметры H и dh
        :param beta: коэффициенты для профиля термодинамической температуры, К.
            Стандартные значения: 6.5 - от 0 до 11 км, 1.0 - от 20 до 32 км, 2.8 - от 32 до 47 км.
        :param HP: характеристическая высота для давления, км
        :param Hrho: характеристическая высота распределения водяного пара, км
        """
        alt = altitudes
        if altitudes is None:
            alt = np.arange(dh, H + dh, dh)

        temperature = []
        T11 = T0 - beta[0] * 11
        T32, T47 = 0., 0.
        for h in alt:
            if h <= 11:
                temperature.append(T0 - beta[0] * h)
            elif 11 < h <= 20:
                temperature.append(T11)
            elif 20 < h <= 32:
                T32 = T11 + (beta[1] * h - 20)
                temperature.append(T32)
            elif 32 < h <= 47:
                T47 = T32 + beta[2] * (h - 32)
                temperature.append(T47)
            else:
                temperature.append(T47)
        temperature = math.as_tensor(temperature)

        pressure = math.as_tensor([P0 * np.exp(-h / HP) for h in alt])

        abs_humidity = math.as_tensor([rho0 * np.exp(-h / Hrho) for h in alt])

        liquid_water = math.zeros_like(abs_humidity)

        if altitudes is None:
            return cls(temperature, pressure, abs_humidity, LiquidWater=liquid_water, dh=dh)
        return cls(temperature, pressure, abs_humidity, LiquidWater=liquid_water, altitudes=altitudes)

    # noinspection PyTypeChecker
    class attenuation:
        """
        Погонные коэффициенты поглощения (ослабления) в зените
        """
        def __init__(self, atmosphere: 'Atmosphere'):
            self.outer = atmosphere

        @atmospheric
        def oxygen(self: 'Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
            """
            См. Rec.ITU-R. P.676-3

            :param frequency: частота излучения в ГГц
            :return: погонный коэффициент поглощения в кислороде (Дб/км)
            """
            return attenuation.oxygen(frequency, self._T, self._P)

        @atmospheric
        def water_vapor(self: 'Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
            """
            См. Rec.ITU-R. P.676-3

            :param frequency: частота излучения в ГГц
            :return: погонный коэффициент поглощения в водяном паре (Дб/км)
            """
            return attenuation.water_vapor(frequency, self._T, self._P, self._rho)

        @atmospheric
        def liquid_water(self: 'Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
            """
            Б.Г. Кутуза

            :param frequency: частота излучения в ГГц
            :return: погонный коэффициент поглощения в облаке (Дб/км)
            """
            if self._use_tcl:
                return attenuation.liquid_water_eff(frequency, self._tcl, self._w)
            return attenuation.liquid_water(frequency, self._T, self._w)

        @atmospheric
        def summary(self: 'Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
            """
            :param frequency: частота излучения в ГГц
            :return: суммарный по атмосферным составляющим погонный коэффициент поглощения (Дб/км)
            """
            return self.attenuation.oxygen(frequency) + self.attenuation.water_vapor(frequency) + \
                self.attenuation.liquid_water(frequency)

    # noinspection PyTypeChecker
    class opacity:
        """
        Расчет полного поглощения атмосферы (оптическая толщина) с учетом угла наблюдения
        """
        def __init__(self, atmosphere: 'Atmosphere'):
            self.outer = atmosphere

        @atmospheric
        def oxygen(self: 'Atmosphere', frequency: float) -> Union[float, Tensor2D]:
            """
            :return: полное поглощение в кислороде (путем интегрирования погонного коэффициента). В неперах
            """
            return dB2np * integrate.full(self.attenuation.oxygen(frequency),
                                          self._dh, self.integration_method,
                                          self._theta, self._PX, self.incline)

        @atmospheric
        def water_vapor(self: 'Atmosphere', frequency: float) -> Union[float, Tensor2D]:
            """
            :return: полное поглощение в водяном паре (путем интегрирования погонного коэффициента). В неперах
            """
            return dB2np * integrate.full(self.attenuation.water_vapor(frequency),
                                          self._dh, self.integration_method,
                                          self._theta, self._PX, self.incline)

        @atmospheric
        def liquid_water(self: 'Atmosphere', frequency: float) -> Union[float, Tensor2D]:
            """
            :return: полное поглощение в облаке (путем интегрирования погонного коэффициента). В неперах
            """
            return dB2np * integrate.full(self.attenuation.liquid_water(frequency),
                                          self._dh, self.integration_method,
                                          self._theta, self._PX, self.incline)

        @atmospheric
        def summary(self: 'Atmosphere', frequency: float) -> Union[float, Tensor2D]:
            """
            :return: полное поглощение в атмосфере (путем интегрирования). В неперах
            """
            return dB2np * integrate.full(self.attenuation.summary(frequency),
                                          self._dh, self.integration_method,
                                          self._theta, self._PX, self.incline)

    # noinspection PyTypeChecker
    class downward:
        """
        Нисходящее излучение
        """
        def __init__(self, atmosphere: 'Atmosphere'):
            self.outer = atmosphere

        @atmospheric
        def brightness_temperature(self: 'Atmosphere', frequency: float,
                                   background=True) -> Union[float, Tensor2D]:
            """
            Яркостная температура нисходящего излучения

            :param frequency: частота излучения в ГГц
            :param background: учитывать космический фон - реликтовое излучение (да/нет)
            """
            g = dB2np * self.attenuation.summary(frequency)
            T = self._T + 273.15

            def f(h):
                integral, b = integrate.limits(g, 0, h, self._dh, self.integration_method,
                                               self._theta, self._PX, self.incline,
                                               boundaries=True)
                return cx(at(T, h), b, h) * cx(at(g, h), b, h) * math.exp(-1 * integral)

            inf = math.len_(g) - 1
            brt, boundaries = integrate.callable_f(f, 0, inf, self._dh, self.integration_method,
                                                   boundaries=True)
            add = 0.
            if background:
                add = self.T_cosmic * math.exp(-1 * self.opacity.summary(frequency))
            return brt + add

    # noinspection PyTypeChecker
    class upward:
        """
        Восходящее излучение
        """
        def __init__(self, atmosphere: 'Atmosphere'):
            self.outer = atmosphere

        @atmospheric
        def brightness_temperature(self: 'Atmosphere', frequency: float) -> Union[float, Tensor2D]:
            """
            Яркостная температура восходящего излучения (без учета подстилающей поверхности)

            :param frequency: частота излучения в ГГц
            """
            g = dB2np * self.attenuation.summary(frequency)
            inf = math.len_(g) - 1
            T = self._T + 273.15

            def f(h):
                integral, b = integrate.limits(g, h, inf, self._dh, self.integration_method,
                                               self._theta, self._PX, self.incline,
                                               boundaries=True)
                return cx(at(T, h), b, h) * cx(at(g, h), b, h) * math.exp(-1 * integral)

            return integrate.callable_f(f, 0, inf, self._dh, self.integration_method)


class avg:
    """
    Расчет средней эффективной температуры атмосферы
    """

    # noinspection PyTypeChecker
    class downward:
        @staticmethod
        def T(sa: Atmosphere, frequency: float) -> Union[float, Tensor2D]:
            """
            Средняя эффективная температура нисходящего излучения атмосферы

            :param frequency: частота излучения в ГГц
            :param sa: стандартная атмосфера (объект Atmosphere)
            """
            tb_down = sa.downward.brightness_temperature(frequency, background=False)
            tau_exp = math.exp(-1 * sa.opacity.summary(frequency))
            return tb_down / (1. - tau_exp)

    # noinspection PyTypeChecker
    class upward:
        @staticmethod
        def T(sa: Atmosphere, frequency: float) -> Union[float, Tensor2D]:
            """
            Средняя эффективная температура восходящего излучения атмосферы

            :param frequency: частота излучения в ГГц
            :param sa: стандартная атмосфера (объект Atmosphere)
            """
            tb_up = sa.upward.brightness_temperature(frequency)
            tau_exp = math.exp(-1 * sa.opacity.summary(frequency))
            return tb_up / (1. - tau_exp)
