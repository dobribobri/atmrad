# -*- coding: utf-8 -*-
from functools import wraps
from multiprocessing import Manager, Process
from typing import Union, Tuple, List, Iterable, Callable
import numpy as np
import warnings

from domain import Domain3D
from cloudforms import CylinderCloud
from planck import Planck


class tf:
    class Tensor:
        pass


TensorLike = Union[np.ndarray, tf.Tensor]
Number = Union[float, complex]
Tensor1D, Tensor2D, Tensor3D, Tensor1D_or_3D = \
    TensorLike, TensorLike, TensorLike, TensorLike

cpu_float = np.float32

C = 299792458
dB2np = 0.23255814
np2dB = 1. / dB2np


def atmospheric(method):
    @wraps(method)
    def wrapper(obj: 'ar.Atmosphere', *args, **kwargs):
        if hasattr(obj, 'outer'):
            obj = obj.outer
        return method(obj, *args, **kwargs)
    return wrapper


def storage(method):
    @wraps(method)
    def wrapper(obj: 'ar.Atmosphere', frequency: float) -> Union[Number, TensorLike]:
        key = (frequency, method.__qualname__)
        if obj.use_storage:
            if key not in obj.storage:
                obj.storage[key] = method(obj, frequency)
            return obj.storage[key]
        return method(obj, frequency)
    return wrapper


# def atmospheric(method):
#     @wraps(method)
#     def wrapper(obj: 'ar.Atmosphere', frequency: float) -> Union[Number, TensorLike]:
#         key = (frequency, method.__qualname__)
#         if hasattr(obj, 'outer'):
#             obj = obj.outer
#         if obj.use_storage:
#             if key not in obj.storage:
#                 obj.storage[key] = method(obj, frequency)
#             return obj.storage[key]
#         return method(obj, frequency)
#     return wrapper


class op_cpu:
    @staticmethod
    def rank(a: TensorLike) -> int:
        return a.ndim

    @staticmethod
    def sum(a: TensorLike, axis: int = None) -> Union[Number, TensorLike]:
        return np.sum(a, axis=axis)

    @staticmethod
    def transpose(a: TensorLike, axes=None) -> TensorLike:
        return np.transpose(a, axes)

    @staticmethod
    def len(a: TensorLike) -> int:
        return a.shape[-1]

    @staticmethod
    def exp(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return np.exp(a)

    @staticmethod
    def log(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return np.log(a)

    @staticmethod
    def sin(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return np.sin(a)

    @staticmethod
    def cos(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return np.cos(a)

    @staticmethod
    def sqrt(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return np.sqrt(a)

    @staticmethod
    def abs(a: Union[Number, TensorLike]) -> Union[float, TensorLike]:
        return np.absolute(a)

    @staticmethod
    def pow(a: Union[Number, TensorLike], d: float) -> Union[float, TensorLike]:
        return np.power(a, d)

    @staticmethod
    def as_tensor(a: Union[Number, TensorLike, Iterable[float]]) -> TensorLike:
        return np.asarray(a, dtype=cpu_float)

    @staticmethod
    def as_variable(a: Union[Number, TensorLike]) -> TensorLike:
        return a

    @staticmethod
    def zeros(shape: Union[int, Iterable[int], Tuple[int]]) -> TensorLike:
        return np.zeros(shape, dtype=cpu_float)

    @staticmethod
    def zeros_like(a: Union[Number, TensorLike, Iterable[float]]) -> TensorLike:
        return np.zeros_like(a, dtype=cpu_float)

    @staticmethod
    def complex(real: float, imag: float = 0.) -> complex:
        return np.complex(real, imag)

    @staticmethod
    def round(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
        return np.round(a)

    @staticmethod
    def mean(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return np.mean(a, axis=axis)

    @staticmethod
    def min(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return np.min(a, axis=axis)

    @staticmethod
    def max(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return np.max(a, axis=axis)

    @staticmethod
    def stddev(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return np.std(a, axis=axis)

    @staticmethod
    def variance(a: TensorLike, axis=None) -> Union[Number, TensorLike]:
        return np.var(a, axis=axis)


class ar:
    op = op_cpu

    class Domain3D(Domain3D):
        pass

    class CylinderCloud(CylinderCloud):
        pass

    class Planck(Planck):
        pass

    class map:
        @staticmethod
        def block_averaging(array: np.ndarray,
                            kernel: Union[Tuple, int] = (10, 10), same_size=True) -> np.ndarray:
            if type(kernel) == int:
                ni = nj = kernel
            else:
                ni, nj = kernel
            if same_size:
                for i in range(0, len(array), ni):
                    for j in range(0, len(array[i]), nj):
                        array[i:i + ni, j:j + nj] = np.mean(array[i:i + ni, j:j + nj])
            else:
                new_arr = np.zeros((len(array) // ni, len(array[0]) // nj), dtype=float)
                for i in range(0, len(array), ni):
                    for j in range(0, len(array[i]), nj):
                        new_arr[i // ni, j // nj] = np.mean(array[i:i + ni, j:j + nj])
                array = new_arr
            return array

        @staticmethod
        def add_zeros(array: np.ndarray, bounds: Union[Tuple[int, int], Tuple[int, int, int, int], int]):
            if type(bounds) == int:
                top = right = bottom = left = bounds
            elif len(bounds) == 2:
                top, right = bounds
                bottom, left = top, right
            elif len(bounds) == 4:
                top, right, bottom, left = bounds
            else:
                raise RuntimeError('неверно задан параметр \'pixels\'')
            assert top >= 0 and right >= 0 and bottom >= 0 and left >= 0, 'только положительные числа'
            h, w = array.shape
            b = np.zeros((h + top + bottom, w + left + right), dtype=array.dtype)
            nh, nw = b.shape
            if bottom == 0:
                bottom = -nh
            if right == 0:
                right = -nw
            b[top:-bottom, left:-right] = array[:, :]
            return b

    class c:

        class indexer:
            @staticmethod
            def at(a: Union[float, Tensor1D_or_3D], index: int) -> Union[Number, Tensor2D]:
                if isinstance(a, float):
                    return a
                rank = ar.op.rank(a)
                if rank not in [1, 3]:
                    raise RuntimeError('неверная размерность')
                if rank == 3:
                    return a[:, :, index]
                return a[index]

            @staticmethod
            def diap(a: Union[float, Tensor1D_or_3D], start: int, stop: int,
                     step: int = 1) -> Union[Number, TensorLike]:
                if isinstance(a, float):
                    return a
                rank = ar.op.rank(a)
                if rank not in [1, 3]:
                    raise RuntimeError('неверная размерность')
                if rank == 3:
                    return a[:, :, start:stop:step]
                return a[start:stop:step]

            @staticmethod
            def last_index(a: TensorLike) -> int:
                return ar.op.len(a) - 1

        class integrate:
            @staticmethod
            def trapz(a: Tensor1D_or_3D, lower: int, upper: int,
                      dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
                return ar.op.sum(ar.c.indexer.diap(a, lower+1, upper) *
                                 ar.c.indexer.diap(dh, lower+1, upper), axis=-1) + \
                       (ar.c.indexer.at(a, lower) * ar.c.indexer.at(dh, lower) +
                        ar.c.indexer.at(a, upper) * ar.c.indexer.at(dh, upper)) / 2.

            @staticmethod
            def simpson(a: Tensor1D_or_3D, lower: int, upper: int,
                        dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
                return (ar.c.indexer.at(a, lower) * ar.c.indexer.at(dh, lower) +
                        ar.c.indexer.at(a, upper) * ar.c.indexer.at(dh, upper) +
                        4 * ar.op.sum(ar.c.indexer.diap(a, lower+1, upper, 2) *
                                      ar.c.indexer.diap(dh, lower+1, upper, 2), axis=-1) +
                        2 * ar.op.sum(ar.c.indexer.diap(a, lower+2, upper, 2) *
                                      ar.c.indexer.diap(dh, lower+2, upper, 2), axis=-1)) / 3.

            @staticmethod
            def boole(a: Tensor1D_or_3D, lower: int, upper: int,
                      dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
                return (14 * (ar.c.indexer.at(a, lower) * ar.c.indexer.at(dh, lower) +
                        ar.c.indexer.at(a, upper) * ar.c.indexer.at(dh, upper)) +
                        64 * ar.op.sum(ar.c.indexer.diap(a, lower+1, upper, 2) *
                                      ar.c.indexer.diap(dh, lower+1, upper, 2), axis=-1) +
                        24 * ar.op.sum(ar.c.indexer.diap(a, lower+2, upper, 4) *
                                      ar.c.indexer.diap(dh, lower+2, upper, 4), axis=-1) +
                        28 * ar.op.sum(ar.c.indexer.diap(a, lower+4, upper, 4) *
                                       ar.c.indexer.diap(dh, lower+4, upper, 4), axis=-1)) / 45.

            @staticmethod
            def limits(a: Tensor1D_or_3D, lower: int, upper: int,
                       dh: Union[float, Tensor1D], method='trapz') -> Union[Number, Tensor2D]:
                if method.lower() == 'trapz':
                    a = ar.c.integrate.trapz(a, lower, upper, dh)
                elif method.lower() == 'simpson':
                    a = ar.c.integrate.simpson(a, lower, upper, dh)
                else:  # boole
                    a = ar.c.integrate.boole(a, lower, upper, dh)
                return a

            @staticmethod
            def full(a: Tensor1D_or_3D, dh: Union[float, Tensor1D], method='trapz') -> Union[Number, Tensor2D]:
                return ar.c.integrate.limits(a, 0, ar.c.indexer.last_index(a), dh, method)

            @staticmethod
            def callable(f: Callable, lower: int, upper: int,
                         dh: Union[float, Tensor1D], method='trapz') -> Union[Number, TensorLike]:
                a = ar.op.as_tensor([f(i) for i in range(lower, upper + 1, 1)])
                a = ar.op.transpose(a, axes=[1, 2, 0])
                return ar.c.integrate.limits(a, lower, upper, dh, method)

            class slantx:
                default_mode = 'valid'

                @staticmethod
                def limits(a: Tensor1D_or_3D, lower: int, upper: int,
                           dh: Union[float, Tensor1D] = 10./500, PX: float = 50.,
                           theta: float = 0., method='trapz',
                           mode=default_mode) -> Union[Number, Tensor2D]:
                    if np.isclose(theta, 0.):
                        return ar.c.integrate.full(a, dh, method)
                    rank = ar.op.rank(a)
                    if rank == 1:
                        return ar.c.integrate.full(a, dh, method) / ar.op.cos(theta)
                    elif rank == 3:
                        Ix, Iy, Iz = a.shape
                        if isinstance(dh, float):
                            dh = ar.op.as_tensor([dh for _ in range(Iz)])
                        dz = dh / ar.op.cos(theta)
                        dx = dz * ar.op.sin(theta)
                        di = Ix * dx / PX
                        assert ar.op.len(di) == Iz, 'размер должен совпадать'
                        full_shift = int(np.round(np.sum(di)))
                        print(full_shift)
                        b = np.zeros((Ix + full_shift, Iy, Iz))
                        for z in range(Iz - 1, -1, -1):
                            shift = int(np.round(np.sum(di[z:])))
                            b[shift:shift + Ix, :, z] = a[:, :, z]
                        integral = ar.c.integrate.limits(ar.op.as_tensor(b), lower, upper, dz, method)
                        if mode.lower() == 'full':
                            return integral
                        if mode.lower() == 'same':
                            return integral[full_shift // 2:full_shift // 2 + Ix, :]  # mode 'same'
                        return integral[full_shift:Ix, :]  # mode 'valid'
                    raise RuntimeError('неверная размерность. Только 1D- и 3D-массивы')

                @staticmethod
                def full(a: Tensor1D_or_3D, dh: Union[float, Tensor1D] = 10./500, PX: float = 50.,
                         theta: float = 0., method='trapz', mode=default_mode) -> Union[Number, Tensor2D]:
                    return ar.c.integrate.slantx.limits(a, 0, ar.c.indexer.last_index(a), dh, PX, theta, method, mode)

                @staticmethod
                def callable(f: Callable, lower: int, upper: int,
                             dh: Union[float, Tensor1D] = 10. / 500, PX: float = 50.,
                             theta: float = 0., method='trapz', mode=default_mode) -> Union[Number, TensorLike]:
                    a = ar.op.as_tensor([f(i) for i in range(lower, upper + 1, 1)])
                    a = ar.op.transpose(a, axes=[1, 2, 0])
                    return ar.c.integrate.slantx.limits(a, lower, upper, dh, PX, theta, method, mode)

        class multi:
            @staticmethod
            def do(processes: list, n_workers: int) -> None:
                for i in range(0, len(processes), n_workers):
                    for j in range(i, i + n_workers):
                        if j < len(processes):
                            processes[j].start()
                    for j in range(i, i + n_workers):
                        if j < len(processes):
                            processes[j].join()

            @staticmethod
            def parallel(frequencies: Union[np.ndarray, List[float]],
                         func: Callable, args: Union[Tuple, List, Iterable],
                         n_workers: int) -> np.ndarray:
                if not n_workers:
                    n_workers = len(frequencies)
                with Manager() as manager:
                    out = manager.list()
                    processes = []
                    for i, f in enumerate(frequencies):
                        p = Process(target=out.append, args=((i, func(f, *args)),))     # TBD
                        # p = Process(target=lambda _out, _i, _f: _out.append((i, func(f, *args))), args=(out, i, f,))
                        processes.append(p)
                    ar.c.multi.do(processes, n_workers)
                    out = list(out)
                return np.asarray([val for _, val in sorted(out, key=lambda item: item[0])], dtype=object)

    class static:

        class water:
            
            class dielectric:
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
                    lambdaS_nosalt = 1.8735116 - 0.027296 * T + 0.000136 * T * T + 1.662 * ar.op.exp(-0.0634 * T)
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
                    return ar.op.complex(eps1, -eps2)

            class Fresnel:
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
                    cos = ar.op.sqrt(epsilon - ar.op.complex(ar.op.cos(psi) * ar.op.cos(psi)))
                    return (ar.op.complex(ar.op.sin(psi)) - cos) / (ar.op.complex(ar.op.sin(psi)) + cos)

                @staticmethod
                def M_vertical(frequency: float, psi: float, T: Union[float, Tensor2D],
                               Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
                    epsilon = ar.static.water.dielectric.epsilon_complex(frequency, T, Sw)
                    cos = ar.op.sqrt(epsilon - ar.op.complex(real=ar.op.cos(psi) * ar.op.cos(psi)))
                    return (epsilon * ar.op.complex(ar.op.sin(psi)) - cos) / \
                           (epsilon * ar.op.complex(ar.op.sin(psi)) + cos)

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
                    val = ar.op.abs(M_h)
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
                    val = ar.op.abs(M_v)
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
                    val = ar.op.abs((ar.op.sqrt(epsilon) - 1) / (ar.op.sqrt(epsilon) + 1))
                    return val * val

            class vapor:
                """
                Водяной пар
                """
                @staticmethod
                def pressure(T: Union[float, TensorLike], rho: Union[float, TensorLike]) -> Union[float, TensorLike]:
                    """
                    Парциальное давление водяного пара

                    :param rho: плотность водяного пара (абсолютная влажность), г/м^3
                    :param T: температура воздуха, град. Цельс.
                    :return: давление в гПа
                    """
                    return rho * (T + 273.15) / 216.7

                @staticmethod
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
                    return ar.static.water.vapor.pressure(T, rho) / \
                        ar.static.water.vapor.saturated.pressure(T, P, method) * 100

                @staticmethod
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
                    return (rel / 100) * 216.7 * ar.static.water.vapor.saturated.pressure(T, P, method) / \
                        (T + 273.15)

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
                            e = 0.61094 * ar.op.exp(17.625 * T / (243.04 + T)) * 10
                        elif method.lower() == 'tetens':
                            e = 0.61078 * ar.op.exp(17.27 * T / (T + 237.3)) * 10
                        elif method.lower() == 'august':
                            e = ar.op.exp(20.386 - 5132 / (T + 273.15)) * 1.333
                        elif method.lower() == 'buck':
                            if T > 0:
                                e = 6.1121 * ar.op.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
                            else:
                                e = 6.1115 * ar.op.exp((23.036 - T / 333.7) * (T / (279.82 + T)))
                        else:
                            e = 6.112 * ar.op.exp(17.62 * T / (243.12 + T))
                        if P is None:
                            return e
                        return (1.0016 + 3.15 * 0.000001 * P - 0.074 / P) * e

        class p676:
            """
            Рекомендация Международного Союза Электросвязи Rec.ITU-R P.676-3
            """
            @staticmethod
            def H1(frequency: float) -> float:
                """
                :param frequency: частота излучения в ГГц
                :return: характеристическая высота поглощения в кислороде (км)
                """
                f = frequency
                const = 6.
                if f < 50:
                    return const
                elif 70 < f < 350:
                    return const + 40 / ((f - 118.7) * (f - 118.7) + 1)
                return const

            @staticmethod
            def H2(frequency: float, rainQ: bool = False) -> float:
                """
                :param frequency: частота излучения в ГГц
                :param rainQ: идет дождь? True/False
                :return: характеристическая высота поглощения в водяном паре (км)
                """
                f = frequency
                Hw = 1.6
                if rainQ:
                    Hw = 2.1
                return Hw * (1 + 3. / ((f - 22.2) * (f - 22.2) + 5) + 5. / ((f - 183.3) * (f - 183.3) + 6) +
                             2.5 / ((f - 325.4) * (f - 325.4) + 4))

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
                f = ar.op.as_tensor(frequency)
                gamma = 0
                if f <= 57:
                    gamma = (7.27 * rt / (f * f + 0.351 * rp * rp * rt * rt) +
                             7.5 / ((f - 57) * (f - 57) + 2.44 * rp * rp * rt * rt * rt * rt * rt)) * \
                            f * f * rp * rp * rt * rt / 1000
                elif 63 <= f <= 350:
                    gamma = (2 / 10000 * ar.op.pow(rt, 1.5) * (1 - 1.2 / 100000 * ar.op.pow(f, 1.5)) +
                             4 / ((f - 63) * (f - 63) + 1.5 * rp * rp * rt * rt * rt * rt * rt) +
                             0.28 * rt * rt / ((f - 118.75) * (f - 118.75) + 2.84 * rp * rp * rt * rt)) * \
                            f * f * rp * rp * rt * rt / 1000
                elif 57 < f < 63:
                    gamma = (f - 60) * (f - 63) / 18 * ar.static.p676.gamma_oxygen(57., T, P) - \
                            1.66 * rp * rp * ar.op.pow(rt, 8.5) * (f - 57) * (f - 63) + \
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
                f = ar.op.as_tensor(frequency)
                gamma = 0
                if f <= 350:
                    gamma = (3.27 / 100 * rt +
                             1.67 / 1000 * rho * rt * rt * rt * rt * rt * rt * rt / rp +
                             7.7 / 10000 * ar.op.pow(f, 0.5) +
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
                return gamma * ar.static.p676.H1(frequency) / ar.op.cos(theta) * dB2np

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
                return gamma * ar.static.p676.H2(frequency, rainQ=rainQ) / ar.op.cos(theta) * dB2np

        class attenuation:
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
            :param altitudes: соответствующие высоты (1D массив), км. Может быть не указан, если указан параметр dh
            :param dh: постоянный шаг по высоте, км. Может быть не указан, если указаны altitudes
            """
            self._T, self._P = Temperature, Pressure
            if AbsoluteHumidity is None:
                AbsoluteHumidity = ar.static.water.vapor.absolute_humidity(self._T, self._P, RelativeHumidity)
            self._rho = AbsoluteHumidity
            assert self._T.shape == self._P.shape == self._rho.shape, 'размерность должна совпадать'
            if altitudes is None and dh is None:
                raise ValueError('пожалуйста, задайте altitudes, либо dh')
            if altitudes is None:
                assert not np.isclose(dh, 0.), 'слишком маленький шаг dh'
                self._dh = dh   # self._dh - float
                self._alt = np.cumsum([dh for _ in range(self._T.shape[-1])], dtype=cpu_float)
            else:
                assert self._T.shape[-1] == len(altitudes), 'altitudes не соответствует Temperature'
                assert not np.isclose(altitudes[0], 0.), 'нулевая высота'
                self._dh = np.diff(np.insert(altitudes, 0, 0.))  # self._dh - np.ndarray
                self._alt = altitudes
            if LiquidWater is None:
                LiquidWater = ar.op.zeros_like(self._T)
            self._w = LiquidWater
            self._tcl = -2  # по Цельсию
            self._theta = 0.  # угол наблюдения в радианах
            self._PX = 50.  # горизонтальная протяженность в километрах
            self.integration_method = 'trapz'
            self.storage: dict = {}
            self.use_storage = True
            for name, value in kwargs.items():
                self.__setattr__(name, value)
            self.attenuation, self.opacity, self.downward, self.upward = \
                None, None, None, None
            self.__interface()

        def __interface(self):
            self.attenuation = ar.Atmosphere.attenuation(self)
            self.opacity = ar.Atmosphere.opacity(self)
            self.downward = ar.Atmosphere.downward(self)
            self.upward = ar.Atmosphere.upward(self)

        def refresh(self, _keys: Union[str, List[str]] = None):
            if _keys is None:
                self.storage.clear()
            for freq, name in list(self.storage.keys()):
                if name in _keys:
                    del self.storage[(freq, name)]

        @property
        def temperature(self) -> Tensor1D_or_3D:
            return self._T

        @temperature.setter
        def temperature(self, val: Tensor1D_or_3D):
            self._T = val
            affected = [self.attenuation.oxygen.__qualname__,
                        self.attenuation.water_vapor.__qualname__,
                        self.attenuation.summary.__qualname__,
                        self.opacity.oxygen.__qualname__,
                        self.opacity.water_vapor.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

        @property
        def pressure(self) -> Tensor1D_or_3D:
            return self._P

        @pressure.setter
        def pressure(self, val: Tensor1D_or_3D):
            self._P = val
            affected = [self.attenuation.oxygen.__qualname__,
                        self.attenuation.water_vapor.__qualname__,
                        self.attenuation.summary.__qualname__,
                        self.opacity.oxygen.__qualname__,
                        self.opacity.water_vapor.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

        @property
        def absolute_humidity(self) -> Tensor1D_or_3D:
            return self._rho

        @absolute_humidity.setter
        def absolute_humidity(self, val: Tensor1D_or_3D):
            self._rho = val
            affected = [self.attenuation.water_vapor.__qualname__,
                        self.attenuation.summary.__qualname__,
                        self.opacity.water_vapor.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

        @property
        def relative_humidity(self) -> Tensor1D_or_3D:
            return ar.static.water.vapor.relative_humidity(self._T, self._P, self._rho)

        @relative_humidity.setter
        def relative_humidity(self, val: Tensor1D_or_3D):
            self.absolute_humidity = ar.static.water.vapor.absolute_humidity(self._T, self._P, val)

        @property
        def liquid_water(self) -> Tensor1D_or_3D:
            return self._w

        @liquid_water.setter
        def liquid_water(self, val: Tensor1D_or_3D):
            self._w = val
            affected = [self.attenuation.liquid_water.__qualname__,
                        self.attenuation.summary.__qualname__,
                        self.opacity.liquid_water.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

        @property
        def altitudes(self) -> np.ndarray:
            return self._alt
        
        @property
        def dh(self) -> Union[float, np.ndarray]:
            return self._dh

        @dh.setter
        def dh(self, val: Union[float, np.ndarray]):
            self._dh = val
            affected = [self.opacity.oxygen.__qualname__,
                        self.opacity.water_vapor.__qualname__,
                        self.opacity.liquid_water.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

        @property
        def effective_cloud_temperature(self) -> float:
            return self._tcl

        @effective_cloud_temperature.setter
        def effective_cloud_temperature(self, val: float):
            self._tcl = val
            affected = [self.attenuation.liquid_water.__qualname__,
                        self.attenuation.summary.__qualname__,
                        self.opacity.liquid_water.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

        @property
        def angle(self) -> float:
            return self._theta

        @angle.setter
        def angle(self, val: float):
            self._theta = val
            affected = [self.opacity.oxygen.__qualname__,
                        self.opacity.water_vapor.__qualname__,
                        self.opacity.liquid_water.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

        @property
        def horizontal_extent(self):
            return self._PX

        @horizontal_extent.setter
        def horizontal_extent(self, val: float):
            self._PX = val
            affected = [self.opacity.oxygen.__qualname__,
                        self.opacity.water_vapor.__qualname__,
                        self.opacity.liquid_water.__qualname__,
                        self.opacity.summary.__qualname__,
                        self.downward.brightness_temperature.__qualname__,
                        self.upward.brightness_temperature.__qualname__]
            self.refresh(affected)

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
            # assert H > 99 * dh, 'H должно быть >> dh'  # !!!
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
            temperature = ar.op.as_tensor(temperature)

            pressure = [P0 * np.exp(-h / HP) for h in altitudes]
            pressure = ar.op.as_tensor(pressure)

            abs_humidity = [rho0 * np.exp(-h / Hrho) for h in altitudes]
            abs_humidity = ar.op.as_tensor(abs_humidity)

            return cls(temperature, pressure, abs_humidity,
                       LiquidWater=ar.op.zeros_like(abs_humidity), dh=dh)

        # noinspection PyTypeChecker
        class attenuation:
            """
            Погонные коэффициенты поглощения (ослабления)
            """
            def __init__(self, atmosphere: 'ar.Atmosphere'):
                self.outer = atmosphere

            @atmospheric
            @storage
            def oxygen(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
                """
                См. Rec.ITU-R. P.676-3

                :param frequency: частота излучения в ГГц
                :return: погонный коэффициент поглощения в кислороде (Дб/км)
                """
                return ar.static.attenuation.oxygen(frequency, self._T, self._P)

            @atmospheric
            @storage
            def water_vapor(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
                """
                См. Rec.ITU-R. P.676-3

                :param frequency: частота излучения в ГГц
                :return: погонный коэффициент поглощения в водяном паре (Дб/км)
                """
                return ar.static.attenuation.water_vapor(frequency, self._T, self._P, self._rho)

            @atmospheric
            @storage
            def liquid_water(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
                """
                Б.Г. Кутуза

                :param frequency: частота излучения в ГГц
                :return: погонный коэффициент поглощения в облаке (Дб/км)
                """
                return ar.static.attenuation.liquid_water(frequency, self._tcl, self._w)

            @atmospheric
            @storage
            def summary(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor1D_or_3D]:
                """
                :param frequency: частота излучения в ГГц
                :return: суммарный по атмосферным составляющим погонный коэффициент поглощения (Дб/км)
                """
                return self.attenuation.oxygen(frequency) + self.attenuation.water_vapor(frequency) + \
                    self.attenuation.liquid_water(frequency)

        # noinspection PyTypeChecker
        class opacity:
            """
            Расчет полного поглощения атмосферы (оптическая толщина)
            """
            def __init__(self, atmosphere: 'ar.Atmosphere'):
                self.outer = atmosphere

            @atmospheric
            @storage
            def oxygen(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в кислороде (путем интегрирования погонного коэффициента). В неперах
                """
                # return dB2np * ar.c.integrate.full(self.attenuation.oxygen(frequency),
                #                                    self._dh, self.integration_method)
                return dB2np * ar.c.integrate.slantx.full(self.attenuation.oxygen(frequency),
                                                          self._dh, self._PX, self._theta, self.integration_method)

            @atmospheric
            @storage
            def water_vapor(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в водяном паре (путем интегрирования погонного коэффициента). В неперах
                """
                # return dB2np * ar.c.integrate.full(self.attenuation.water_vapor(frequency),
                #                                    self._dh, self.integration_method)
                return dB2np * ar.c.integrate.slantx.full(self.attenuation.water_vapor(frequency),
                                                          self._dh, self._PX, self._theta, self.integration_method)

            @atmospheric
            @storage
            def liquid_water(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в облаке (путем интегрирования погонного коэффициента). В неперах
                """
                # return dB2np * ar.c.integrate.full(self.attenuation.liquid_water(frequency),
                #                                    self._dh, self.integration_method)
                return dB2np * ar.c.integrate.slantx.full(self.attenuation.liquid_water(frequency),
                                                          self._dh, self._PX, self._theta, self.integration_method)

            @atmospheric
            @storage
            def summary(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                :return: полное поглощение в атмосфере (путем интегрирования). В неперах
                """
                # return dB2np * ar.c.integrate.full(self.attenuation.summary(frequency),
                #                                    self._dh, self.integration_method)
                return dB2np * ar.c.integrate.slantx.full(self.attenuation.summary(frequency),
                                                          self._dh, self._PX, self._theta, self.integration_method)

        class downward:
            """
            Нисходящее излучение
            """
            def __init__(self, atmosphere: 'ar.Atmosphere'):
                self.outer = atmosphere

            @atmospheric
            @storage
            def brightness_temperature(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Яркостная температура нисходящего излучения

                :param frequency: частота излучения в ГГц
                """
                g = dB2np * self.attenuation.summary(frequency)
                T = self._T + 273.15
                f = lambda h: ar.c.indexer.at(T, h) * ar.c.indexer.at(g, h) * \
                    ar.op.exp(-1 * ar.c.integrate.limits(g, 0, h, self._dh, self.integration_method))
                inf = ar.c.indexer.last_index(g)
                # return ar.c.integrate.callable(f, 0, inf, self._dh)
                return ar.c.integrate.slantx.callable(f, 0, inf,
                                                      self._dh, self._PX, self._theta, self.integration_method)

            @atmospheric
            def brightness_temperatures(self: 'ar.Atmosphere', frequencies: Union[np.ndarray, List[float]],
                                        n_workers: int = None) -> np.ndarray:
                """
                Яркостная температура нисходящего излучения

                :param frequencies: список частот в ГГц
                :param n_workers: количество потоков для распараллеливания
                """
                # if self.use_storage:
                #     warnings.warn('storage не может быть использован')
                return ar.c.multi.parallel(frequencies,
                                           func=self.downward.brightness_temperature,
                                           args=(), n_workers=n_workers)

        class upward:
            """
            Восходящее излучение
            """
            def __init__(self, atmosphere: 'ar.Atmosphere'):
                self.outer = atmosphere

            @atmospheric
            @storage
            def brightness_temperature(self: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Яркостная температура восходящего излучения (без учета подстилающей поверхности)

                :param frequency: частота излучения в ГГц
                """
                g = dB2np * self.attenuation.summary(frequency)
                inf = ar.c.indexer.last_index(g)
                T = self._T + 273.15
                f = lambda h: ar.c.indexer.at(T, h) * ar.c.indexer.at(g, h) * \
                    ar.op.exp(-1 * ar.c.integrate.limits(g, h, inf, self._dh, self.integration_method))
                # return ar.c.integrate.callable(f, 0, inf, self._dh)
                return ar.c.integrate.slantx.callable(f, 0, inf,
                                                      self._dh, self._PX, self._theta, self.integration_method)

            @atmospheric
            def brightness_temperatures(self: 'ar.Atmosphere', frequencies: Union[np.ndarray, List[float]],
                                        n_workers: int = None) -> np.ndarray:
                """
                Яркостная температура восходящего излучения (без учета подстилающей поверхности)

                :param frequencies: список частот в ГГц
                :param n_workers: количество потоков для распараллеливания
                """
                # if self.use_storage:
                #     warnings.warn('storage не может быть использован')
                return ar.c.multi.parallel(frequencies,
                                           func=self.upward.brightness_temperature,
                                           args=(), n_workers=n_workers)

    # noinspection PyArgumentList
    class weight_functions:
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
            return tau_water_vapor / (ar.c.integrate.full(sa.absolute_humidity, sa.dh) / 10.)

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

        @staticmethod
        def Staelin(sa: 'ar.Atmosphere', frequency: float) -> Tensor1D_or_3D:
            """
            :param frequency: частота излучения в ГГц
            :param sa: стандартная атмосфера (объект Atmosphere)
            :return: весовая функция Стилина
            """
            return sa.attenuation.water_vapor(frequency) / sa.absolute_humidity

    class avg:
        """
        Расчет средней эффективной температуры атмосферы
        """
        # noinspection PyArgumentList
        class downward:
            @staticmethod
            def T(sa: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Средняя эффективная температура нисходящего излучения атмосферы

                :param frequency: частота излучения в ГГц
                :param sa: стандартная атмосфера (объект Atmosphere)
                """
                tb_down = sa.downward.brightness_temperature(frequency)
                tau_exp = ar.op.exp(-1 * sa.opacity.summary(frequency))
                return tb_down / (1. - tau_exp)

        # noinspection PyArgumentList
        class upward:
            @staticmethod
            def T(sa: 'ar.Atmosphere', frequency: float) -> Union[float, Tensor2D]:
                """
                Средняя эффективная температура восходящего излучения атмосферы

                :param frequency: частота излучения в ГГц
                :param sa: стандартная атмосфера (объект Atmosphere)
                """
                tb_up = sa.upward.brightness_temperature(frequency)
                tau_exp = ar.op.exp(-1 * sa.opacity.summary(frequency))
                return tb_up / (1. - tau_exp)

    class Surface:
        def __init__(self, surface_temperature: Union[float, Tensor2D],
                     theta: float = 0.,
                     polarization: str = None, **kwargs):
            self._T = surface_temperature
            self._theta = theta
            self._polarization = polarization
            for name, value in kwargs.items():
                self.__setattr__(name, value)

        @property
        def temperature(self) -> Union[float, Tensor2D]:
            return self._T

        @temperature.setter
        def temperature(self, val: Union[float, Tensor2D]):
            self._T = val

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
        def __init__(self, surface_temperature: Union[float, Tensor2D] = 15.,
                     salinity: Union[float, Tensor2D] = 0.,
                     theta: float = 0., polarization: str = None):
            """
            :param surface_temperature: термодинамическая температура поверхности, град. Цельс.
            :param salinity: соленость, промили
            :param theta: зенитный угол, рад.
            :param polarization: поляризация ('H' или 'V')
            """
            super().__init__(surface_temperature, theta, polarization)
            self._Sw = salinity

        @property
        def salinity(self) -> Union[float, Tensor2D]:
            return self._Sw

        @salinity.setter
        def salinity(self, val: Union[float, Tensor2D]):
            self._Sw = val

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

        def emissivity(self, frequency: float) -> Union[float, Tensor2D]:
            """
            Расчет излучательной способности

            :param frequency: частота излучения в ГГц
            :return: коэффициент излучения гладкой водной поверхности при условии термодинамического равновесия
            """
            return 1. - self.reflectivity(frequency)

    # noinspection PyArgumentList
    class satellite:
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
            tau_exp = ar.op.exp(-1 * atm.opacity.summary(frequency))
            tb_down = atm.downward.brightness_temperature(frequency)
            tb_up = atm.upward.brightness_temperature(frequency)
            r = srf.reflectivity(frequency)
            kappa = 1. - r  # emissivity
            return (srf.temperature + 273.15) * kappa * tau_exp + tb_up + r * tb_down * tau_exp

        class multi:
            @staticmethod
            def brightness_temperature(frequencies: Union[np.ndarray, List[float]], atm: 'ar.Atmosphere',
                                       srf: 'ar.Surface', n_workers: int = None) -> np.ndarray:
                """
                Яркостная температура уходящего излучения системы 'атмосфера - подстилающая поверхность'

                :param frequencies: список частот в ГГц
                :param atm: объект Atmosphere (атмосфера)
                :param srf: объект Surface (поверхность)
                :param n_workers: количество потоков для распараллеливания
                """
                # if atm.use_storage:
                #     warnings.warn('storage не может быть использован')
                return ar.c.multi.parallel(frequencies,
                                           func=ar.satellite.brightness_temperature,
                                           args=(atm, srf,), n_workers=n_workers)

    class inverse:
        pass


if __name__ == '__main__':
    pass
