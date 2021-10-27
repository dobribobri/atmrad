
# -*- coding: utf-8 -*-
from typing import Union, Tuple, List, Iterable, Callable
import numpy as np
from cpu import ar as cpu
from cpu import TensorLike, Number
from cpu import Tensor1D, Tensor2D, Tensor1D_or_3D
from cpu import C, dB2np, np2dB
from cpu import atmospheric

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

gpu_float = tf.float32


class ar(cpu):

    class _c(cpu._c):

        @staticmethod
        def rank(a: TensorLike) -> int:
            return tf.rank(a)

        @staticmethod
        def sum(a: TensorLike, axis: int = None) -> Union[Number, TensorLike]:
            return tf.reduce_sum(a, axis=axis)

        @staticmethod
        def transpose(a: TensorLike, axes=None) -> TensorLike:
            return tf.transpose(a, perm=axes)

        @staticmethod
        def len(a: TensorLike) -> int:
            return tf.shape(a)[-1]

        @staticmethod
        def exp(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            return tf.exp(a)

        @staticmethod
        def log(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            return tf.math.log(a)

        @staticmethod
        def sin(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            return tf.sin(a)

        @staticmethod
        def cos(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            return tf.cos(a)

        @staticmethod
        def sqrt(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            return tf.sqrt(a)

        @staticmethod
        def abs(a: Union[Number, TensorLike]) -> Union[float, TensorLike]:
            return tf.abs(a)

        @staticmethod
        def pow(a: Union[Number, TensorLike], d: float) -> Union[float, TensorLike]:
            return tf.pow(a, d)

        @staticmethod
        def as_tensor(a: Union[Number, TensorLike, Iterable[float]]) -> TensorLike:
            return tf.cast(tf.convert_to_tensor(a), dtype=gpu_float)

        @staticmethod
        def zeros_like(a: Union[Number, TensorLike, Iterable[float]]) -> TensorLike:
            return tf.zeros_like(a, dtype=gpu_float)

        class indexer(cpu._c.indexer):
            @staticmethod
            def at(a: Tensor1D_or_3D, index: int) -> Union[Number, Tensor2D]:
                rank = ar._c.rank(a)
                if rank not in [1, 3]:
                    raise RuntimeError('неверная размерность')
                if rank == 3:
                    return a[:, :, index]
                return a[index]

            @staticmethod
            def last_index(a: TensorLike) -> int:
                return ar._c.len(a) - 1

        class integrate(cpu._c.integrate):
            @staticmethod
            def _trapz(a: Tensor1D_or_3D, lower: int, upper: int,
                       dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
                if isinstance(dh, float):
                    return ar._c.sum(a[lower + 1:upper], axis=0) * dh + (a[lower] + a[upper]) / 2. * dh
                return ar._c.sum(a[lower + 1:upper] * dh[lower + 1:upper], axis=0) + \
                    (a[lower] * dh[lower] + a[upper] * dh[upper]) / 2.

            @staticmethod
            def _simpson(a: Tensor1D_or_3D, lower: int, upper: int,
                         dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
                if isinstance(dh, float):
                    return (a[lower] + a[upper] + 4 * ar._c.sum(a[lower + 1:upper:2], axis=0) +
                            2 * ar._c.sum(a[lower + 2:upper:2], axis=0)) * dh / 3.
                return (a[lower] * dh[lower] + a[upper] * dh[upper] +
                        4 * ar._c.sum(a[lower + 1:upper:2] * dh[lower + 1:upper:2], axis=0) +
                        2 * ar._c.sum(a[lower + 2:upper:2] * dh[lower + 2:upper:2], axis=0)) / 3.

            @staticmethod
            def _boole(a: Tensor1D_or_3D, lower: int, upper: int,
                       dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
                if isinstance(dh, float):
                    return (14 * (a[lower] + a[upper]) + 64 * ar._c.sum(a[lower + 1:upper:2], axis=0) +
                            24 * ar._c.sum(a[lower + 2:upper:4], axis=0) +
                            28 * ar._c.sum(a[lower + 4:upper:4], axis=0)) * dh / 45.
                return (14 * (a[lower] * dh[lower] + a[upper] * dh[upper]) +
                        64 * ar._c.sum(a[lower + 1:upper:2] * dh[lower + 1:upper:2], axis=0) +
                        24 * ar._c.sum(a[lower + 2:upper:4] * dh[lower + 2:upper:4], axis=0) +
                        28 * ar._c.sum(a[lower + 4:upper:4] * dh[lower + 4:upper:4], axis=0)) / 45.

            @staticmethod
            def with_limits(a: Tensor1D_or_3D, lower: int, upper: int,
                            dh: Union[float, Tensor1D], method='trapz') -> Union[Number, Tensor2D]:
                if method not in ['trapz', 'simpson', 'boole']:
                    raise ValueError('выберите один из доступных методов: \'trapz\', \'simpson\', \'boole\'')
                rank = ar._c.rank(a)
                if rank not in [1, 3]:
                    raise RuntimeError('неверная размерность. Только 1D- и 3D-массивы')
                if rank == 3:
                    a = ar._c.transpose(a, [2, 0, 1])
                if method == 'trapz':
                    a = ar._c.integrate._trapz(a, lower, upper, dh)
                if method == 'simpson':
                    a = ar._c.integrate._simpson(a, lower, upper, dh)
                if method == 'boole':
                    a = ar._c.integrate._boole(a, lower, upper, dh)
                return a

            @staticmethod
            def full(a: Tensor1D_or_3D, dh: Union[float, Tensor1D], method='trapz') -> Union[Number, Tensor2D]:
                return ar._c.integrate.with_limits(a, 0, ar._c.indexer.last_index(a), dh, method)

        class multi:
            @staticmethod
            def parallel(frequencies: Union[np.ndarray, List[float]],
                         func: Callable, args: Union[Tuple, List, Iterable]) -> np.ndarray:
                return np.asarray([func(f, *args) for f in frequencies], dtype=object)

    class static(cpu.static):

        class water(cpu.static.water):

            class dielectric(cpu.static.water.dielectric):
                """
                Диэлектрическая проницаемость воды с учетом солености
                """
                @staticmethod
                def epsilon(T: Union[float, TensorLike],
                            Sw: Union[float, TensorLike] = 0.) -> Tuple[float, float, float]:
                    """
                    :param T: термодинамическая температура воды, град. Цельс.
                    :param Sw: соленость, промили
                    :return: кортеж значений: 1 - оптическая составляющая диэлектрической проницаемости,
                        2 - статическая составляющая, 3 - характерная длина волны
                    """
                    epsO_nosalt = 5.5
                    epsS_nosalt = 88.2 - 0.40885 * T + 0.00081 * T * T
                    lambdaS_nosalt = 1.8735116 - 0.027296 * T + 0.000136 * T * T + 1.662 * ar._c.exp(-0.0634 * T)
                    epsO = epsO_nosalt
                    epsS = epsS_nosalt - 17.2 * Sw / 60
                    lambdaS = lambdaS_nosalt - 0.206 * Sw / 60
                    return epsO, epsS, lambdaS

                @staticmethod
                def epsilon_complex(frequency: float, T: Union[float, TensorLike],
                                    Sw: Union[float, TensorLike] = 0.) -> Union[complex, TensorLike]:
                    """
                    Комплексная диэлектрическая проницаемость воды

                    :param frequency: частота излучения в ГГц
                    :param T: термодинамическая температура воды, град. Цельс.
                    :param Sw: соленость, промили
                    """
                    lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
                    epsO, epsS, lambdaS = ar.static.water.dielectric.epsilon(T, Sw)
                    y = lambdaS / lamda
                    eps1 = epsO + (epsS - epsO) / (1 + y * y)
                    eps2 = y * (epsS - epsO) / (1 + y * y)
                    sigma = 0.00001 * (2.63 * T + 77.5) * Sw
                    eps2 = eps2 + 60 * sigma * lamda
                    return tf.complex(eps1, -eps2)

            class Fresnel(cpu.static.water.Fresnel):
                """
                Формулы Френеля
                """
                @staticmethod
                def M_horizontal(frequency: float, psi: float, T: Union[float, Tensor2D],
                                 Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
                    """

                    :param frequency: частота излучения в ГГц
                    :param psi: угол скольжения, рад.
                    :param T: температура поверхности, град. Цельс.
                    :param Sw: соленость, промили
                    """
                    epsilon = ar.static.water.dielectric.epsilon_complex(frequency, T, Sw)
                    cos = ar._c.sqrt(epsilon - ar._c.cos(psi) * ar._c.cos(psi))
                    return (ar._c.sin(psi) - cos) / (ar._c.sin(psi) + cos)

                @staticmethod
                def M_vertical(frequency: float, psi: float, T: Union[float, Tensor2D],
                               Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
                    epsilon = ar.static.water.dielectric.epsilon_complex(frequency, T, Sw)
                    cos = ar._c.sqrt(epsilon - ar._c.cos(psi) * ar._c.cos(psi))
                    return (epsilon * ar._c.sin(psi) - cos) / (epsilon * ar._c.sin(psi) + cos)

                @staticmethod
                def R_horizontal(frequency: float, theta: float, T: Union[float, Tensor2D],
                                 Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
                    """
                    :param frequency: частота излучения в ГГц
                    :param theta: зенитный угол, рад.
                    :param T: температура поверхности, град. Цельс.
                    :param Sw: соленость, промили
                    :return: коэффициент отражения на горизонтальной поляризации
                    """
                    M_h = ar.static.water.Fresnel.M_horizontal(frequency, np.pi / 2. - theta, T, Sw)
                    val = ar._c.abs(M_h)
                    return val * val

                @staticmethod
                def R_vertical(frequency: float, theta: float, T: Union[float, Tensor2D],
                               Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
                    """
                    :param frequency: частота излучения в ГГц
                    :param theta: зенитный угол, рад.
                    :param T: температура поверхности, град. Цельс.
                    :param Sw: соленость, промили
                    :return: коэффициент отражения на вертикальной поляризации
                    """
                    M_v = ar.static.water.Fresnel.M_vertical(frequency, np.pi / 2. - theta, T, Sw)
                    val = ar._c.abs(M_v)
                    return val * val

                @staticmethod
                def R(frequency: float, T: Union[float, Tensor2D],
                      Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
                    """
                    :param frequency: частота излучения в ГГц
                    :param T: температура поверхности, град. Цельс.
                    :param Sw: соленость, промили
                    :return: коэффициент отражения при зенитном угле 0 рад
                    """
                    epsilon = ar.static.water.dielectric.epsilon_complex(frequency, T, Sw)
                    val = ar._c.abs((ar._c.sqrt(epsilon) - 1) / (ar._c.sqrt(epsilon) + 1))
                    return val * val

        class p676(cpu.static.p676):
            """
            Рекомендация Международного Союза Электросвязи Rec.ITU-R P.676-3
            """
            @staticmethod
            def gamma_oxygen(frequency: float,
                             T: Union[float, TensorLike], P: Union[float, TensorLike]) -> Union[float, TensorLike]:
                """
                :param frequency: частота излучения в ГГц
                :param T: термодинамическая температура, градусы Цельсия
                :param P: атмосферное давление, мбар или гПа
                :return: погонный коэффициент поглощения в кислороде (Дб/км)
                """
                rp = P / 1013
                rt = 288 / (273 + T)
                f = tf.convert_to_tensor(frequency, dtype=gpu_float)
                gamma = 0
                if f <= 57:
                    gamma = (7.27 * rt / (f * f + 0.351 * rp * rp * rt * rt) +
                             7.5 / ((f - 57) * (f - 57) + 2.44 * rp * rp * rt * rt * rt * rt * rt)) * \
                            f * f * rp * rp * rt * rt / 1000
                elif 63 <= f <= 350:
                    gamma = (2 / 10000 * ar._c.pow(rt, 1.5) * (1 - 1.2 / 100000 * ar._c.pow(f, 1.5)) +
                             4 / ((f - 63) * (f - 63) + 1.5 * rp * rp * rt * rt * rt * rt * rt) +
                             0.28 * rt * rt / ((f - 118.75) * (f - 118.75) + 2.84 * rp * rp * rt * rt)) * \
                            f * f * rp * rp * rt * rt / 1000
                elif 57 < f < 63:
                    gamma = (f - 60) * (f - 63) / 18 * ar.static.p676.gamma_oxygen(57., T, P) - \
                            1.66 * rp * rp * ar._c.pow(rt, 8.5) * (f - 57) * (f - 63) + \
                            (f - 57) * (f - 60) / 18 * ar.static.p676.gamma_oxygen(63., T, P)
                return gamma

            @staticmethod
            def gamma_water_vapor(frequency: float,
                                  T: Union[float, TensorLike], P: Union[float, TensorLike],
                                  rho: Union[float, TensorLike]) -> Union[float, TensorLike]:
                """
                :param frequency: частота излучения в ГГц
                :param T: термодинамическая температура, градусы Цельсия
                :param P: атмосферное давление, мбар или гПа
                :param rho: абсолютная влажность, г/м^3
                :return: погонный коэффициент поглощения в водяном паре (Дб/км)
                """
                rp = P / 1013
                rt = 288 / (273 + T)
                f = tf.convert_to_tensor(frequency, dtype=gpu_float)
                gamma = 0
                if f <= 350:
                    gamma = (3.27 / 100 * rt +
                             1.67 / 1000 * rho * rt * rt * rt * rt * rt * rt * rt / rp +
                             7.7 / 10000 * ar._c.pow(f, 0.5) +
                             3.79 / ((f - 22.235) * (f - 22.235) + 9.81 * rp * rp * rt) +
                             11.73 * rt / ((f - 183.31) * (f - 183.31) + 11.85 * rp * rp * rt) +
                             4.01 * rt / ((f - 325.153) * (f - 325.153) + 10.44 * rp * rp * rt)) * \
                            f * f * rho * rp * rt / 10000
                return gamma

            @staticmethod
            def tau_oxygen_near_ground(frequency: float,
                                       T_near_ground: Union[float, Tensor2D],
                                       P_near_ground: Union[float, Tensor2D],
                                       theta: float = 0.0) -> Union[float, Tensor2D]:
                """
                Учитывает угол наблюдения.

                :param frequency: частота излучения в ГГц
                :param T_near_ground: значение или 2D-срез температуры приземного слоя воздуха, градусы Цельсия
                :param P_near_ground: значение или 2D-срез атмосферного давления, гПа
                :param theta: угол наблюдения в радианах
                :return: полное поглощение в кислороде (оптическая толщина). В неперах
                """
                gamma = ar.static.p676.gamma_oxygen(frequency, T_near_ground, P_near_ground)
                return gamma * ar.static.p676.H1(frequency) / ar._c.cos(theta) * dB2np

            @staticmethod
            def tau_water_vapor_near_ground(frequency: float,
                                            T_near_ground: Union[float, Tensor2D],
                                            P_near_ground: Union[float, Tensor2D],
                                            rho_near_ground: Union[float, Tensor2D],
                                            theta: float = 0.0, rainQ=False) -> Union[float, Tensor2D]:
                """
                Учитывает угол наблюдения.

                :param frequency: частота излучения в ГГц
                :param T_near_ground: значение или 2D-срез температуры приземного слоя воздуха, градусы Цельсия
                :param P_near_ground: значение или 2D-срез приповерхностного атмосферного давления, гПа
                :param rho_near_ground: значение или 2D-срез приповерхностной абсолютной влажности, г/м^3
                :param theta: угол наблюдения в радианах
                :param rainQ: идет дождь? True/False
                :return: полное поглощение в водяном паре. В неперах
                """
                gamma = ar.static.p676.gamma_water_vapor(frequency, T_near_ground, P_near_ground, rho_near_ground)
                return gamma * ar.static.p676.H2(frequency, rainQ=rainQ) / ar._c.cos(theta) * dB2np

        class attenuation(cpu.static.attenuation):
            """
            Погонные коэффициенты поглощения (ослабления)
            """
            @staticmethod
            def oxygen(frequency: float,
                       T: Union[float, TensorLike], P: Union[float, TensorLike]) -> Union[float, TensorLike]:
                """
                Копия static.p676.gamma_oxygen(...)

                :param frequency: частота излучения в ГГц
                :param T: термодинамическая температура, градусы Цельсия
                :param P: атмосферное давление, мбар или гПа
                :return: погонный коэффициент поглощения в кислороде (Дб/км)
                """
                return ar.static.p676.gamma_oxygen(frequency, T, P)

            @staticmethod
            def water_vapor(frequency: float,
                            T: Union[float, TensorLike], P: Union[float, TensorLike],
                            rho: Union[float, TensorLike]) -> Union[float, TensorLike]:
                """
                Копия static.p676.gamma_water_vapor(...)

                :param frequency: частота излучения в ГГц
                :param T: термодинамическая температура, градусы Цельсия
                :param P: атмосферное давление, мбар или гПа
                :param rho: абсолютная влажность, г/м^3
                :return: погонный коэффициент поглощения в водяном паре (Дб/км)
                """
                return ar.static.p676.gamma_water_vapor(frequency, T, P, rho)

            @staticmethod
            def liquid_water(frequency: float, t_cloud: float,
                             w: Union[float, TensorLike]) -> Union[float, TensorLike]:
                """
                Б.Г. Кутуза

                :param frequency: частота излучения в ГГц
                :param t_cloud: средняя эффективная температура облаков, град. Цельс.
                :param w: поле водности, кг/м^3
                :return: погонный коэффициент поглощения в облаке (Дб/км)
                """
                return np2dB * ar.weight_functions.kw_(frequency, t_cloud) * w

    class Atmosphere(cpu.Atmosphere):

        def __interface(self):
            self.attenuation = ar.Atmosphere.attenuation(self)
            self.opacity = ar.Atmosphere.opacity(self)
            self.downward = ar.Atmosphere.downward(self)
            self.upward = ar.Atmosphere.upward(self)

        @classmethod
        def Standard(cls, T0: float = 15., P0: float = 1013, rho0: float = 7.5,
                     H: float = 10, dh: float = 10. / 500,
                     beta: Tuple[float, float, float] = (6.5, 1., 2.8),
                     HP: float = 7.7, Hrho: float = 2.1) -> 'ar.Atmosphere':
            """
            Стандарт атмосферы

            :param T0: приповерхностная температура, град. Цельс.
            :param P0: давление на уровне поверхности, гПа
            :param rho0: приповерхностное значение абсолютной влажности, г/м^3
            :param H: высота расчетной области, км
            :param dh: шаг по высоте, км
            :param beta: коэффициенты для профиля термодинамической температуры, К.
                Стандартные значения: 6.5 - от 0 до 11 км, 1.0 - от 20 до 32 км, 2.8 - от 32 до 47 км.
            :param HP: характеристическая высота для давления, км
            :param Hrho: характеристическая высота распределения водяного пара, км
            """
            assert H > 99 * dh, 'H должно быть >> dh'
            altitudes = np.arange(dh, H + dh, dh)

            temperature = []
            T11 = T0 - beta[0] * 11
            T32, T47 = 0., 0.
            for h in altitudes:
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
            temperature = ar._c.as_tensor(temperature)

            pressure = [P0 * np.exp(-h / HP) for h in altitudes]
            pressure = ar._c.as_tensor(pressure)

            abs_humidity = [rho0 * np.exp(-h / Hrho) for h in altitudes]
            abs_humidity = ar._c.as_tensor(abs_humidity)

            return cls(temperature, pressure, abs_humidity,
                       LiquidWater=ar._c.zeros_like(abs_humidity), dh=dh)

        # noinspection PyTypeChecker
        class attenuation(cpu.Atmosphere.attenuation):
            """
            Погонные коэффициенты поглощения (ослабления)
            """
            @atmospheric
            def oxygen(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
                """
                См. Rec.ITU-R. P.676-3

                :param frequency: частота излучения в ГГц
                :return: погонный коэффициент поглощения в кислороде (Дб/км)
                """
                return ar.static.attenuation.oxygen(frequency, self._T, self._P)

            @atmospheric
            def water_vapor(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
                """
                См. Rec.ITU-R. P.676-3

                :param frequency: частота излучения в ГГц
                :return: погонный коэффициент поглощения в водяном паре (Дб/км)
                """
                return ar.static.attenuation.water_vapor(frequency, self._T, self._P, self._rho)

            @atmospheric
            def liquid_water(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
                """
                Б.Г. Кутуза

                :param frequency: частота излучения в ГГц
                :return: погонный коэффициент поглощения в облаке (Дб/км)
                """
                return ar.static.attenuation.liquid_water(frequency, self._tcl, self._w)

        # noinspection PyTypeChecker
        class opacity(cpu.Atmosphere.opacity):
            """
            Расчет полного поглощения атмосферы (оптическая толщина)
            """
            @atmospheric
            def oxygen(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в кислороде (путем интегрирования погонного коэффициента). В неперах
                """
                return dB2np * ar._c.integrate.full(self.attenuation.oxygen(frequency),
                                                    self._dh, self.integration_method)

            @atmospheric
            def water_vapor(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в водяном паре (путем интегрирования погонного коэффициента). В неперах
                """
                return dB2np * ar._c.integrate.full(self.attenuation.water_vapor(frequency),
                                                    self._dh, self.integration_method)

            @atmospheric
            def liquid_water(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в облаке (путем интегрирования погонного коэффициента). В неперах
                """
                return dB2np * ar._c.integrate.full(self.attenuation.liquid_water(frequency),
                                                    self._dh, self.integration_method)

            @atmospheric
            def summary(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в атмосфере (путем интегрирования). В неперах
                """
                return dB2np * ar._c.integrate.full(self.attenuation.summary(frequency),
                                                    self._dh, self.integration_method)

        class downward(cpu.Atmosphere.downward):
            """
            Нисходящее излучение
            """
            @atmospheric
            def brightness_temperature(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Яркостная температура нисходящего излучения

                :param frequency: частота излучения в ГГц
                """
                g = dB2np * self.attenuation.summary(frequency)
                T = self._T + 273.15
                f = lambda h: ar._c.indexer.at(T, h) * ar._c.indexer.at(g, h) * \
                    ar._c.exp(-1 * ar._c.integrate.with_limits(g, 0, h, self._dh, self.integration_method))
                inf = ar._c.indexer.last_index(g)
                return ar._c.integrate.callable(f, 0, inf, self._dh)

            def brightness_temperatures(self: 'ar.Atmosphere', frequencies: Union[np.ndarray, List[float]],
                                        n_workers: int = None) -> np.ndarray:
                """
                Яркостная температура нисходящего излучения

                :param frequencies: список частот в ГГц
                :param n_workers: количество потоков для распараллеливания
                    (в режиме gpu игнорируется)
                """
                return ar._c.multi.parallel(frequencies,
                                            func=self.downward.brightness_temperature,
                                            args=())

        class upward(cpu.Atmosphere.upward):
            """
            Восходящее излучение
            """
            @atmospheric
            def brightness_temperature(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Яркостная температура восходящего излучения (без учета подстилающей поверхности)

                :param frequency: частота излучения в ГГц
                """
                g = dB2np * self.attenuation.summary(frequency)
                inf = ar._c.indexer.last_index(g)
                T = self._T + 273.15
                f = lambda h: ar._c.indexer.at(T, h) * ar._c.indexer.at(g, h) * \
                    ar._c.exp(-1 * ar._c.integrate.with_limits(g, h, inf, self._dh, self.integration_method))
                return ar._c.integrate.callable(f, 0, inf, self._dh)

            def brightness_temperatures(self: 'ar.Atmosphere', frequencies: Union[np.ndarray, List[float]],
                                        n_workers: int = None) -> np.ndarray:
                """
                Яркостная температура восходящего излучения (без учета подстилающей поверхности)

                :param frequencies: список частот в ГГц
                :param n_workers: количество потоков для распараллеливания
                    (в режиме gpu игнорируется)
                """
                return ar._c.multi.parallel(frequencies,
                                            func=self.upward.brightness_temperature,
                                            args=())

    # noinspection PyArgumentList
    class weight_functions(cpu.weight_functions):
        """
        Различные весовые функции
        """
        @staticmethod
        def krho(sa: 'ar.Atmosphere', frequency: float) -> float:
            """
            :param frequency: частота излучения в ГГц
            :param sa: стандартная атмосфера (объект Atmosphere)
            :return: весовая функция krho (водяной пар)
            """
            tau_water_vapor = sa.opacity.water_vapor(frequency)
            return tau_water_vapor / (ar._c.integrate.full(sa.absolute_humidity, sa.dh) / 10.)

        @staticmethod
        def kw_(frequency: float, t_cloud: float) -> float:
            """
            :param frequency: частота излучения в ГГц
            :param t_cloud: средняя эффективная температура облака, град. Цельс.
            :return: весовая функция k_w (вода в жидкокапельной фазе).
            """
            lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
            epsO, epsS, lambdaS = ar.static.water.dielectric.epsilon(t_cloud, 0.)
            y = lambdaS / lamda
            return 3 * 0.6 * np.pi / lamda * (epsS - epsO) * y / (
                    (epsS + 2) * (epsS + 2) + (epsO + 2) * (epsO + 2) * y * y)

        @staticmethod
        def kw(sa: 'ar.Atmosphere', frequency: float) -> float:
            """
            :param frequency: частота излучения в ГГц
            :param sa: объект Atmosphere
            :return: весовая функция k_w (вода в жидкокапельной фазе).
            """
            return ar.weight_functions.kw_(frequency, sa.effective_cloud_temperature)

    class avg(cpu.avg):
        """
        Расчет средней эффективной температуры атмосферы
        """
        # noinspection PyArgumentList
        class downward(cpu.avg.downward):
            @staticmethod
            def T(sa: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Средняя эффективная температура нисходящего излучения атмосферы

                :param frequency: частота излучения в ГГц
                :param sa: стандартная атмосфера (объект Atmosphere)
                """
                tb_down = sa.downward.brightness_temperature(frequency)
                tau_exp = ar._c.exp(-1 * sa.opacity.summary(frequency))
                return tb_down / (1. - tau_exp)

        # noinspection PyArgumentList
        class upward(cpu.avg.upward):
            @staticmethod
            def T(sa: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Средняя эффективная температура восходящего излучения атмосферы

                :param frequency: частота излучения в ГГц
                :param sa: стандартная атмосфера (объект Atmosphere)
                """
                tb_up = sa.upward.brightness_temperature(frequency)
                tau_exp = ar._c.exp(-1 * sa.opacity.summary(frequency))
                return tb_up / (1. - tau_exp)

    class Surface(cpu.Surface):
        pass

    class SmoothWaterSurface(cpu.SmoothWaterSurface):
        """
        Модель микроволнового излучения гладкой водной поверхности
        """
        def reflectivity(self, frequency: float) -> Union[float, Tensor2D]:
            """
            Расчет отражательной способности

            :param frequency: частота излучения в ГГц
            :return: коэффициент отражения гладкой водной поверхности
            """
            if np.isclose(self._theta, 0.):
                return ar.static.water.Fresnel.R(frequency, self._T, self._Sw)
            if self._polarization in ['H', 'h']:
                return ar.static.water.Fresnel.R_horizontal(frequency, self._theta, self._T, self._Sw)
            return ar.static.water.Fresnel.R_vertical(frequency, self._theta, self._T, self._Sw)

    # noinspection PyArgumentList
    class satellite(cpu.satellite):
        """
        Спутник
        """
        @staticmethod
        def brightness_temperature(frequency: float, atm: 'ar.Atmosphere',
                                   srf: 'ar.Surface') -> Union[float, Tensor2D]:
            """
            Яркостная температура уходящего излучения системы 'атмосфера - подстилающая поверхность'

            :param frequency: частота излучения в ГГц
            :param atm: объект Atmosphere (атмосфера)
            :param srf: объект Surface (поверхность)
            """
            tau = atm.opacity.summary(frequency)
            tau_exp = ar._c.exp(-1 * tau)
            tb_down = atm.downward.brightness_temperature(frequency)
            tb_up = atm.upward.brightness_temperature(frequency)
            r = srf.reflectivity(frequency)
            kappa = 1. - r  # emissivity
            return (srf.temperature + 273.15) * kappa * tau_exp + tb_up + r * tb_down * tau_exp

        class multi:
            @staticmethod
            def brightness_temperature(frequencies: Union[np.ndarray, List[float]], atm: 'ar.Atmosphere',
                                       srf: 'ar.Surface') -> np.ndarray:
                """
                Яркостная температура уходящего излучения системы 'атмосфера - подстилающая поверхность'

                :param frequencies: список частот в ГГц
                :param atm: объект Atmosphere (атмосфера)
                :param srf: объект Surface (поверхность)
                """
                return ar._c.multi.parallel(frequencies,
                                            func=ar.satellite.brightness_temperature,
                                            args=(atm, srf,))

    class inverse:
        pass
