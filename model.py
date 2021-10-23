# -*- coding: utf-8 -*-

from typing import Union, List, Iterable, Tuple, Callable, Any
from multiprocessing import Manager, Process
# from collections import defaultdict
from scipy.special import gamma
import numpy as np
from functools import partial
import time
# import re

from cloudforms import CylinderCloud
from domain import Domain
from decor import storage
import settings
gpu = settings.gpu

if gpu:
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
else:
    from tf_stub import *

cpu_float = np.float32
gpu_float = tf.float32
cpu_types = (np.ndarray, float, cpu_float, complex)

C = 299792458
dB2np = 0.23255814
np2dB = 1. / dB2np

TensorLike = Union[np.ndarray, tf.Tensor]
Number = Union[float, complex]
Tensor1D, Tensor2D, Tensor3D, Tensor1D_or_3D = \
    TensorLike, TensorLike, TensorLike, TensorLike


class Standard:
    def __init__(self, T0: float = 15. + 273.15, P0: float = 1013, rho0: float = 7.5,
                 H: float = 40, dh: float = 40. / 500,
                 nx: int = None, ny: int = None,
                 near_ground=False):
        """
        :param T0: К
        :param P0: гПа
        :param rho0: г/м^3
        :param H: км
        :param dh: км
        :param nx: безразм.
        :param ny: безразм.
        :param near_ground: если True, H и dh не учитываются.
        Все массивы возвращаются только для уровня H = 0 (приземный слой).
        """
        self.T0, self.P0, self.rho0 = T0, P0, rho0
        self.H, self.dh = H, dh
        self.nx, self.ny = nx, ny
        self.near_ground = near_ground

    def set(self, **kwargs) -> None:
        for name, value in kwargs:
            self.__setattr__(name, value)

    @property
    def __altitudes(self) -> np.ndarray:
        if self.near_ground:
            return np.asarray([0.])
        return np.arange(0, self.H + self.dh / 2, self.dh)

    @staticmethod
    def __(profile: Union[np.ndarray, List[float], Iterable[float]]) -> Union[float, np.ndarray]:
        if len(profile) == 1:
            return profile[0]
        return np.asarray(profile, dtype=cpu_float)

    @staticmethod
    def reshape(profile: Union[float, np.ndarray], sizes: tuple) -> Union[float, np.ndarray]:
        nx, ny = sizes
        if nx and ny:
            profile = np.asarray([[profile for _ in range(ny)] for _ in range(nx)], dtype=cpu_float)
        return profile

    @staticmethod
    def gpu(profile: Union[float, np.ndarray]) -> Union[float, Tensor1D_or_3D]:
        if gpu:
            return tf.convert_to_tensor(profile, dtype=gpu_float)
        return profile

    def altitudes(self) -> Union[float, Tensor1D]:
        return self.gpu(self.__(self.__altitudes))

    def zeros(self) -> Union[float, Tensor1D_or_3D]:
        return self.gpu(self.reshape(self.__(np.zeros(len(self.__altitudes), dtype=cpu_float)), (self.nx, self.ny)))

    def temperature(self, celsius=True) -> Union[float, Tensor1D_or_3D]:
        """
        :param celsius: град. Цельс.?
        :return: Стандартный высотный профиль термодинамической температуры
        (К или Цельс. в зависимости от celsius).
        """
        profile = []
        T11 = self.T0 - 6.5 * 11
        T32, T47 = 0., 0.
        for h in self.__altitudes:
            if h <= 11:
                profile.append(self.T0 - 6.5 * h)
            elif 11 < h <= 20:
                profile.append(T11)
            elif 20 < h <= 32:
                T32 = T11 + (1 * h - 20)
                profile.append(T32)
            elif 32 < h <= 47:
                T47 = T32 + 2.8 * (h - 32)
                profile.append(T47)
            else:
                profile.append(T47)
        profile = self.__(profile)
        if celsius:
            profile -= 273.15
        return self.gpu(self.reshape(profile, (self.nx, self.ny)))

    def T(self, celsius=True) -> Union[float, Tensor1D_or_3D]:
        return self.temperature(celsius)

    def pressure(self, HP: float = 7.7) -> Union[float, Tensor1D_or_3D]:
        """
        :return: Стандартный высотный профиль атмосферного давления (гПа).
        """
        profile = self.__([self.P0 * np.exp(-h / HP) for h in self.__altitudes])
        return self.gpu(self.reshape(profile, (self.nx, self.ny)))

    def P(self, HP: float = 7.7) -> Union[float, Tensor1D_or_3D]:
        return self.pressure(HP)

    def absolute_humidity(self, Hrho: float = 2.1) -> Union[float, Tensor1D_or_3D]:
        """
        :return: Стандартный высотный профиль абсолютной влажности (г/м^3).
        """
        profile = self.__([self.rho0 * np.exp(-h / Hrho) for h in self.__altitudes])
        return self.gpu(self.reshape(profile, (self.nx, self.ny)))

    def rho(self, Hrho: float = 2.1) -> Union[float, Tensor1D_or_3D]:
        return self.absolute_humidity(Hrho)


class Planck(Domain):
    def __init__(self, kilometers: Tuple[float, float, float] = None,
                 nodes: Tuple[Union[float, None], Union[float, None], float] = None):
        super().__init__(kilometers, nodes)

    def create_height_map(self, Dm: float, K: float,
                          alpha: float, beta: float, eta: float, seed: int = 0,
                          verbose=True) -> np.ndarray:
        np.random.seed(seed)
        cloudiness = []
        r = np.sqrt(self.i(Dm) * self.i(Dm) + self.j(Dm) * self.j(Dm))
        steps = np.arange(Dm, 0, -Dm / r)
        N = len(steps)
        for i, D in enumerate(steps):
            if verbose:
                print('\r{:.2f}%'.format((i + 1) / N * 100), end='', flush=True)
            n = int(np.round(K * np.exp(-alpha * D)))
            if n < 1:
                n = 1
            for k in range(n):
                start_time = time.time()
                while True:
                    x = np.random.uniform(0., self.PX)
                    y = np.random.uniform(0., self.PY)
                    z = self.cl_bottom
                    rx = D / 2
                    ry = D / 2
                    H = eta * D * np.power(D / Dm, beta)
                    cloud = CylinderCloud((x, y, z), rx, ry, H)
                    if not cloud.belongsQ((self.PX, self.PY, self.PZ)):
                        continue
                    intersections = False
                    for c in cloudiness:
                        if not cloud.disjointQ(c):
                            intersections = True
                            break
                    if not intersections:
                        cloudiness.append(cloud)
                        break
                    if time.time() - start_time > 30:
                        raise TimeoutError('превышено допустимое время ожидания')
        print()
        hmap = np.zeros((self.Nx, self.Ny))
        for cloud in cloudiness:
            for x in np.arange(cloud.x - cloud.rx, cloud.x + cloud.rx, self.x(1)):
                for y in np.arange(cloud.y - cloud.ry, cloud.y + cloud.ry, self.y(1)):
                    if cloud.includesQ((x, y, self.cl_bottom)):
                        hmap[self.i(x), self.j(y)] = cloud.height
        return np.asarray(hmap, dtype=cpu_float)

    def liquid_water_distribution(self, height_map: np.ndarray, const_w=False,
                                  mu0: float = 3.27, psi0: float = 0.67) -> Tensor3D:
        min_level = self.k(self.cl_bottom)
        max_level = self.k(self.cl_bottom + np.max(height_map))
        if gpu:
            hmap = tf.convert_to_tensor(height_map, dtype=gpu_float)
            W = 0.132574 * tf.pow(hmap, 2.30215)
        else:
            hmap = np.asarray(height_map, dtype=cpu_float)
            W = 0.132574 * np.power(hmap, 2.30215)

        if gpu:
            w = tf.Variable(tf.zeros((self.Nx, self.Ny, self.Nz), dtype=gpu_float))
            cond = tf.logical_not(tf.experimental.numpy.isclose(hmap, 0.))
            zeros, ones = tf.zeros_like(hmap), tf.ones_like(hmap)
            if const_w:
                for k in range(min_level, max_level):
                    xi = tf.where(cond, (self.z(k) - self.cl_bottom) / hmap, zeros)
                    xi = tf.where(tf.logical_or(tf.greater(0., xi), tf.greater(xi, 1.)), zeros, ones)
                    w[:, :, k].assign(tf.where(cond, xi * W / hmap, zeros))
            else:
                for k in range(min_level, max_level):
                    xi = tf.where(cond, (self.z(k) - self.cl_bottom) / hmap, zeros)
                    xi = tf.where(tf.logical_or(tf.greater(0., xi), tf.greater(xi, 1.)), zeros, xi)
                    w[:, :, k].assign(tf.where(cond,
                                               tf.pow(xi, mu0) * tf.pow(1 - xi, psi0) * W / hmap *
                                               gamma(2 + mu0 + psi0) / (gamma(1 + mu0) * gamma(1 + psi0)),
                                               zeros))
        else:
            w = np.zeros((self.Nx, self.Ny, self.Nz))
            cond = np.logical_not(np.isclose(hmap, 0.))
            if const_w:
                for k in range(min_level, max_level):
                    xi = (self.z(k) - self.cl_bottom) / hmap[cond]
                    xi[(xi < 0) | (xi > 1)] = 0.
                    xi[(0 <= xi) | (xi <= 1)] = 1.
                    w[cond, k] = xi * W[cond] / hmap[cond]
            else:
                for k in range(min_level, max_level):
                    xi = (self.z(k) - self.cl_bottom) / hmap[cond]
                    xi[(xi < 0) | (xi > 1)] = 0.
                    w[cond, k] = \
                        np.power(xi, mu0) * np.power(1 - xi, psi0) * W[cond] / hmap[cond] * \
                        gamma(2 + mu0 + psi0) / (gamma(1 + mu0) * gamma(1 + psi0))

        if gpu:
            return tf.convert_to_tensor(w, dtype=gpu_float)
        return np.asarray(w, dtype=cpu_float)


class Model:
    def __init__(self, T_std: Tensor1D = None, P_std: Tensor1D = None, rho_std: Tensor1D = None,
                 altitudes: Tensor1D = None, dh: float = None):

        if altitudes is None and dh is None:
            raise ValueError('пожалуйста, задайте altitudes, либо dh')
        if altitudes is None:
            self.dh = dh
        else:
            dh = [altitudes[0]]
            for i in range(1, len(altitudes)):
                dh.append(altitudes[i] - altitudes[i - 1])
            if gpu:
                self.dh = tf.convert_to_tensor(dh, dtype=gpu_float)
            else:
                self.dh = np.asarray(dh, dtype=cpu_float)
        sp = Standard()
        if T_std is None:
            self.T_std = sp.T(celsius=True)
        else:
            self.T_std = T_std
        if P_std is None:
            self.P_std = sp.P()
        else:
            self.P_std = P_std
        if rho_std is None:
            self.rho_std = sp.rho()
        else:
            self.rho_std = rho_std
        assert Model._compat.are_numpy_arrays(self.T_std, self.P_std, self.rho_std) or \
            Model._compat.are_tensors(self.T_std, self.P_std, self.rho_std), 'типы должны совпадать'

        self._T, self._P, self._rho = self.T_std, self.P_std, self.rho_std
        self._w, self._t_cloud = 0., -2.
        self._T_ocean, self._Sw_ocean = 15., 0.
        self._TK = 0.
        # self._storage = defaultdict(Any)
        self._storage = {}

    def refresh(self, _what: Union[str, List[str], Iterable[str]] = None):
        if _what is None:
            self._storage.clear()
        for freq, name in self._storage.keys():
            if name in _what:
                del self._storage[(freq, name)]

    @property
    def T(self):
        return self._T

    @property
    def P(self):
        return self._P

    @property
    def rho(self):
        return self._rho

    @property
    def w(self):
        return self._w

    @property
    def t_cloud(self):
        return self._t_cloud

    @property
    def Tsurface_ocean(self):
        return self._T_ocean

    @property
    def Salinity_ocean(self):
        return self._Sw_ocean

    @property
    def T_cosmic(self):
        return self._TK

    @T.setter
    def T(self, val: Union[float, Tensor1D_or_3D]):
        self._T = val
        affected = [self.gamma_oxygen.__name__, self.gamma_water_vapor.__name__, self.gamma_summary.__name__,
                    self.tau_oxygen.__name__, self.tau_water_vapor.__name__, self.tau_summary.__name__,
                    self.brightness_temperature_downward.__name__,
                    self.brightness_temperature_upward.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    @P.setter
    def P(self, val: Union[float, Tensor1D_or_3D]):
        self._P = val
        affected = [self.gamma_oxygen.__name__, self.gamma_water_vapor.__name__, self.gamma_summary.__name__,
                    self.tau_oxygen.__name__, self.tau_water_vapor.__name__, self.tau_summary.__name__,
                    self.brightness_temperature_downward.__name__,
                    self.brightness_temperature_upward.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    @rho.setter
    def rho(self, val: Union[float, Tensor1D_or_3D]):
        self._rho = val
        affected = [self.gamma_oxygen.__name__, self.gamma_water_vapor.__name__, self.gamma_summary.__name__,
                    self.tau_oxygen.__name__, self.tau_water_vapor.__name__, self.tau_summary.__name__,
                    self.brightness_temperature_downward.__name__,
                    self.brightness_temperature_upward.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    @w.setter
    def w(self, val: Union[float, Tensor3D]):
        self._w = val
        affected = [self.gamma_liquid_water.__name__, self.tau_liquid_water.__name__,
                    self.gamma_summary.__name__, self.tau_summary.__name__,
                    self.brightness_temperature_downward.__name__,
                    self.brightness_temperature_upward.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    @t_cloud.setter
    def t_cloud(self, val: float):
        self._t_cloud = val
        affected = [self.kw.__name__,
                    self.gamma_liquid_water.__name__, self.tau_liquid_water.__name__,
                    self.gamma_summary.__name__, self.tau_summary.__name__,
                    self.brightness_temperature_downward.__name__,
                    self.brightness_temperature_upward.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    @Tsurface_ocean.setter
    def Tsurface_ocean(self, val: Union[float, Tensor2D]):
        self._T_ocean = val
        affected = [self.smooth_water_reflectance.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    @Salinity_ocean.setter
    def Salinity_ocean(self, val: Union[float, Tensor2D]):
        self._Sw_ocean = val
        affected = [self.smooth_water_reflectance.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    @T_cosmic.setter
    def T_cosmic(self, val: Union[float, Tensor2D]):
        self._TK = val
        affected = [self.brightness_temperature_downward.__name__,
                    self.T_avg_downward.__name__,
                    self.brightness_temperature_satellite.__name__]
        self.refresh(affected)

    def set(self, T: Union[float, Tensor1D_or_3D] = None, P: Union[float, Tensor1D_or_3D] = None,
            rho: Union[float, Tensor1D_or_3D] = None, w: Union[float, Tensor3D] = None,
            Tsurface_ocean: Union[float, Tensor2D] = None, Salinity_ocean: Union[float, Tensor2D] = None,
            t_cloud: float = None, T_cosmic: float = None):
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P
        if rho is not None:
            self.rho = rho
        if w is not None:
            self.w = w
        if Tsurface_ocean is not None:
            self.Tsurface_ocean = Tsurface_ocean
        if Salinity_ocean is not None:
            self.Salinity_ocean = Salinity_ocean
        if t_cloud is not None:
            self.t_cloud = t_cloud
        if T_cosmic is not None:
            self.T_cosmic = T_cosmic

    class _compat:
        @staticmethod
        def rank(a: TensorLike) -> int:
            if isinstance(a, cpu_types):
                return a.ndim
            return tf.rank(a)

        @staticmethod
        def sum(a: TensorLike, axis: int = None) -> Union[Number, TensorLike]:
            if isinstance(a, cpu_types):
                return np.sum(a, axis=axis)
            return tf.reduce_sum(a, axis=axis)

        @staticmethod
        def transpose(a: TensorLike, axes=None) -> TensorLike:
            if isinstance(a, cpu_types):
                return np.transpose(a, axes)
            return tf.transpose(a, perm=axes)

        @staticmethod
        def len(a: TensorLike) -> int:
            if isinstance(a, cpu_types):
                return a.shape[-1]
            return tf.shape(a)[-1]

        @staticmethod
        def last_index(a: TensorLike) -> int:
            return Model._compat.len(a) - 1

        @staticmethod
        def _trapz(a: Tensor1D_or_3D, lower: int, upper: int,
                   dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
            if isinstance(dh, float):
                return Model._compat.sum(a[lower+1:upper], axis=0) * dh + (a[lower] + a[upper]) / 2. * dh
            return Model._compat.sum(a[lower+1:upper] * dh[lower+1:upper], axis=0) + \
                (a[lower] * dh[lower] + a[upper] * dh[upper]) / 2.

        @staticmethod
        def _simpson(a: Tensor1D_or_3D, lower: int, upper: int,
                     dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
            if isinstance(dh, float):
                return (a[lower] + a[upper] + 4 * Model._compat.sum(a[lower+1:upper:2], axis=0) +
                        2 * Model._compat.sum(a[lower+2:upper:2], axis=0)) * dh / 3.
            return (a[lower] * dh[lower] + a[upper] * dh[upper] +
                    4 * Model._compat.sum(a[lower+1:upper:2] * dh[lower+1:upper:2], axis=0) +
                    2 * Model._compat.sum(a[lower+2:upper:2] * dh[lower+2:upper:2], axis=0)) / 3.

        @staticmethod
        def _boole(a: Tensor1D_or_3D, lower: int, upper: int,
                   dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
            if isinstance(dh, float):
                return (14 * (a[lower] + a[upper]) + 64 * Model._compat.sum(a[lower + 1:upper:2], axis=0) +
                    24 * Model._compat.sum(a[lower + 2:upper:4], axis=0) +
                    28 * Model._compat.sum(a[lower + 4:upper:4], axis=0)) * dh / 45.
            return (14 * (a[lower] * dh[lower] + a[upper] * dh[upper]) +
                    64 * Model._compat.sum(a[lower+1:upper:2] * dh[lower+1:upper:2], axis=0) +
                    24 * Model._compat.sum(a[lower+2:upper:4] * dh[lower+2:upper:4], axis=0) +
                    28 * Model._compat.sum(a[lower+4:upper:4] * dh[lower+4:upper:4], axis=0)) / 45.

        @staticmethod
        def integrate_bounds(a: Tensor1D_or_3D, lower: int, upper: int,
                             dh: Union[float, Tensor1D], method='trapz') -> Union[Number, Tensor2D]:
            if method not in ['trapz', 'simpson', 'boole']:
                raise ValueError('выберите один из доступных методов: \'trapz\', \'simpson\', \'boole\'')
            rank = Model._compat.rank(a)
            if rank not in [1, 3]:
                raise RuntimeError('неверная размерность')
            if rank == 3:
                a = Model._compat.transpose(a, [2, 0, 1])
            if method == 'trapz':
                a = Model._compat._trapz(a, lower, upper, dh)
            if method == 'simpson':
                a = Model._compat._simpson(a, lower, upper, dh)
            if method == 'boole':
                a = Model._compat._boole(a, lower, upper, dh)
            return a

        @staticmethod
        def integrate(a: Tensor1D_or_3D,
                      dh: Union[float, Tensor1D], method='trapz') -> Union[Number, Tensor2D]:
            return Model._compat.integrate_bounds(a, 0, Model._compat.last_index(a), dh, method)

        @staticmethod
        def integrate_callable(f: Callable, lower: int, upper: int,
                               dh: Union[float, Tensor1D]) -> Union[Number, TensorLike]:
            if isinstance(dh, float):
                a = dh * (f(lower) + f(upper)) / 2.
                for k in range(lower + 1, upper):
                    a += dh * f(k)
                return a
            a = (dh[lower] * f(lower) + dh[upper] * f(upper)) / 2.
            for k in range(lower + 1, upper):
                a += dh[k] * f(k)
            return a

        @staticmethod
        def exp(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            if isinstance(a, cpu_types):
                return np.exp(a)
            return tf.exp(a)

        @staticmethod
        def log(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            if isinstance(a, cpu_types):
                return np.log(a)
            return tf.math.log(a)

        @staticmethod
        def sin(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            if isinstance(a, cpu_types):
                return np.sin(a)
            return tf.sin(a)

        @staticmethod
        def cos(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            if isinstance(a, cpu_types):
                return np.cos(a)
            return tf.cos(a)

        @staticmethod
        def sqrt(a: Union[Number, TensorLike]) -> Union[Number, TensorLike]:
            if isinstance(a, cpu_types):
                return np.sqrt(a)
            return tf.sqrt(a)

        @staticmethod
        def abs(a: Union[Number, TensorLike]) -> Union[float, TensorLike]:
            if isinstance(a, cpu_types):
                return np.absolute(a)
            return tf.abs(a)

        @staticmethod
        def to_numpy_array(a: Union[Number, TensorLike], dtype: Any = cpu_float) -> np.ndarray:
            if not (isinstance(a, cpu_types)):
                a = a.numpy()
            return np.asarray(a, dtype=dtype)

        @staticmethod
        def to_tensor(a: Union[Number, TensorLike], dtype: Any = gpu_float) -> tf.Tensor:
            return tf.cast(tf.convert_to_tensor(a), dtype=dtype)

        @staticmethod
        def are_numpy_arrays(*arrays) -> bool:
            for arr in arrays:
                if not isinstance(arr, np.ndarray):
                    return False
            return True

        @staticmethod
        def are_tensors(*arrays) -> bool:
            for arr in arrays:
                if not (isinstance(arr, tf.Tensor) or isinstance(arr, tf.Variable)):
                    return False
            return True

        @staticmethod
        def at(a: TensorLike, index: int) -> Union[Number, TensorLike]:
            rank = Model._compat.rank(a)
            if rank not in [1, 3]:
                raise RuntimeError('неверная размерность')
            if rank == 3:
                return a[:, :, index]
            return a[index]

    class _multi:
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
        def parallel(frequencies: Union[np.ndarray, List[float], Iterable[float]],
                     func: Callable, args: Union[Tuple, List, Iterable],
                     n_workers: int) -> np.ndarray:
            if not n_workers:
                n_workers = len(frequencies)
            with Manager() as manager:
                integrals = manager.list()
                processes = []
                for i, f in enumerate(frequencies):
                    p = Process(target=lambda _integrals, _i, _freq:
                    _integrals.append((_i, func(_freq, *args))),
                                args=(integrals, i, f,))
                    processes.append(p)
                Model._multi.do(processes, n_workers)
                integrals = list(integrals)
            return np.asarray([integral for _, integral in
                               sorted(integrals, key=lambda item: item[0])], dtype=object)

    class p676:
        @staticmethod
        def H1(frequency: float) -> float:
            """
            :return: характеристическая поглощения в кислороде (км)
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
            :return: характеристическая поглощения в водяном паре (км)
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
            :return: погонный коэффициент поглощения в кислороде (Дб/км)
            """
            rp = P / 1013
            rt = 288 / (273 + T)
            f = frequency
            gamma = 0
            if f <= 57:
                gamma = (7.27 * rt / (f * f + 0.351 * rp * rp * rt * rt) +
                         7.5 / ((f - 57) * (f - 57) + 2.44 * rp * rp * rt * rt * rt * rt * rt)) * \
                        f * f * rp * rp * rt * rt / 1000
            elif 63 <= f <= 350:
                gamma = (2 / 10000 * np.power(rt, 1.5) * (1 - 1.2 / 100000 * np.power(f, 1.5)) +
                         4 / ((f - 63) * (f - 63) + 1.5 * rp * rp * rt * rt * rt * rt * rt) +
                         0.28 * rt * rt / ((f - 118.75) * (f - 118.75) + 2.84 * rp * rp * rt * rt)) * \
                        f * f * rp * rp * rt * rt / 1000
            elif 57 < f < 63:
                gamma = (f - 60) * (f - 63) / 18 * Model.p676.gamma_oxygen(57., T, P) - \
                        1.66 * rp * rp * np.power(rt, 8.5) * (f - 57) * (f - 63) + \
                        (f - 57) * (f - 60) / 18 * Model.p676.gamma_oxygen(63., T, P)
            return gamma

        @staticmethod
        def gamma_water_vapor(frequency: float,
                              T: Union[float, TensorLike], P: Union[float, TensorLike],
                              rho: Union[float, TensorLike]) -> Union[float, TensorLike]:
            """
            :return: погонный коэффициент поглощения в водяном паре (Дб/км)
            """
            rp = P / 1013
            rt = 288 / (273 + T)
            f = frequency
            gamma = 0
            if f <= 350:
                gamma = (3.27 / 100 * rt +
                         1.67 / 1000 * rho * rt * rt * rt * rt * rt * rt * rt / rp +
                         7.7 / 10000 * np.power(f, 0.5) +
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

            :return: полное поглощение в кислороде (модельн.). В неперах.
            """
            gamma = Model.p676.gamma_oxygen(frequency, T_near_ground, P_near_ground)
            return gamma * Model.p676.H1(frequency) / np.cos(theta) * dB2np

        @staticmethod
        def tau_water_vapor_near_ground(frequency: float,
                                        T_near_ground: Union[float, Tensor2D],
                                        P_near_ground: Union[float, Tensor2D],
                                        rho_near_ground: Union[float, Tensor2D],
                                        theta: float = 0.0, rainQ=False) -> Union[float, Tensor2D]:
            """
            Учитывает угол наблюдения.

            :return: полное поглощение в водяном паре (модельн.). В неперах
            """
            gamma = Model.p676.gamma_water_vapor(frequency, T_near_ground, P_near_ground, rho_near_ground)
            return gamma * Model.p676.H2(frequency, rainQ=rainQ) / np.cos(theta) * dB2np

    class dielectric:
        @staticmethod
        def epsilon(T: Union[float, Tensor2D],
                    Sw: Union[float, Tensor2D] = 0.) -> Tuple[float, float, float]:
            epsO_nosalt = 5.5
            epsS_nosalt = 88.2 - 0.40885 * T + 0.00081 * T * T
            lambdaS_nosalt = 1.8735116 - 0.027296 * T + 0.000136 * T * T + \
                1.662 * Model._compat.exp(-0.0634 * T)
            epsO = epsO_nosalt
            epsS = epsS_nosalt - 17.2 * Sw / 60
            lambdaS = lambdaS_nosalt - 0.206 * Sw / 60
            return epsO, epsS, lambdaS

        @staticmethod
        def epsilon_complex(frequency: float, T: Union[float, Tensor2D],
                            Sw: Union[float, Tensor2D] = 0.) -> Union[complex, Tensor2D]:
            lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
            epsO, epsS, lambdaS = Model.dielectric.epsilon(T, Sw)
            y = lambdaS / lamda
            eps1 = epsO + (epsS - epsO) / (1 + y * y)
            eps2 = y * (epsS - epsO) / (1 + y * y)
            sigma = 0.00001 * (2.63 * T + 77.5) * Sw
            eps2 = eps2 + 60 * sigma * lamda
            return eps1 - 1j * eps2

    class weight_func:
        @staticmethod
        def krho(frequency: float,
                 T_standard: Tensor1D,
                 P_standard: Tensor1D,
                 rho_standard: Tensor1D, dh: Union[float, Tensor1D]) -> float:
            gamma = Model.p676.gamma_water_vapor(frequency, T_standard, P_standard, rho_standard)
            tau_water_vapor = dB2np * Model._compat.integrate(gamma, dh)
            return tau_water_vapor / (Model._compat.integrate(rho_standard, dh) / 10.)

        @staticmethod
        def kw(frequency: float, t_cloud: float) -> float:
            """
            Не учитывает угол наблюдения. В неперах.

            :param frequency: частота излучения
            :param t_cloud: средняя эффективная температура облака
            :return: весовая функция k_w (вода в жидкокапельной фазе).
            """
            lamda = C / (frequency * 10 ** 9) * 100  # перевод в [cm]
            epsO, epsS, lambdaS = Model.dielectric.epsilon(t_cloud)
            y = lambdaS / lamda
            return 3 * 0.6 * np.pi / lamda * (epsS - epsO) * y / (
                    (epsS + 2) * (epsS + 2) + (epsO + 2) * (epsO + 2) * y * y)

    @storage
    def krho(self, frequency: float) -> float:
        return Model.weight_func.krho(frequency, self.T_std, self.P_std, self.rho_std, self.dh)

    @storage
    def kw(self, frequency: float) -> float:
        return Model.weight_func.kw(frequency, self.t_cloud)

    class attenuation:
        @staticmethod
        def gamma_oxygen(frequency: float,
                         T: Union[float, TensorLike], P: Union[float, TensorLike]) -> Union[float, TensorLike]:
            return Model.p676.gamma_oxygen(frequency, T, P)

        @staticmethod
        def gamma_water_vapor(frequency: float,
                              T: Union[float, TensorLike], P: Union[float, TensorLike],
                              rho: Union[float, TensorLike]) -> Union[float, TensorLike]:
            return Model.p676.gamma_water_vapor(frequency, T, P, rho)

        @staticmethod
        def gamma_liquid_water(frequency: float, t_cloud: float,
                               w: Union[float, Tensor3D]) -> Union[float, Tensor3D]:
            if isinstance(w, (float, np.ndarray)):
                return np2dB * Model.weight_func.kw(frequency, t_cloud) * w
            return Model._compat.to_tensor(np2dB * Model.weight_func.kw(frequency, t_cloud)) * w

        @staticmethod
        def gamma_summary(frequency: float,
                          T: Union[float, Tensor1D_or_3D], P: Union[float, Tensor1D_or_3D],
                          rho: Union[float, Tensor1D_or_3D], w: Union[float, Tensor3D] = None,
                          t_cloud: float = None, nepers=False) -> Union[float, Tensor3D]:
            k = 1.
            if nepers:
                k = dB2np
            if w is None or t_cloud is None:
                return k * (Model.attenuation.gamma_oxygen(frequency, T, P) +
                            Model.attenuation.gamma_water_vapor(frequency, T, P, rho))
            return k * (Model.attenuation.gamma_oxygen(frequency, T, P) +
                        Model.attenuation.gamma_water_vapor(frequency, T, P, rho) +
                        Model.attenuation.gamma_liquid_water(frequency, t_cloud, w))

    @storage
    def gamma_oxygen(self, frequency: float) -> Union[float, Tensor1D_or_3D]:
        return Model.attenuation.gamma_oxygen(frequency, self.T, self.P)

    @storage
    def gamma_water_vapor(self, frequency: float) -> Union[float, Tensor1D_or_3D]:
        return Model.attenuation.gamma_water_vapor(frequency, self.T, self.P, self.rho)

    @storage
    def gamma_liquid_water(self, frequency: float) -> Union[float, Tensor3D]:
        return Model.attenuation.gamma_liquid_water(frequency, self.t_cloud, self.w)

    @storage
    def gamma_summary(self, frequency: float) -> Union[float, Tensor3D]:    # в децибелах
        return Model.attenuation.gamma_summary(frequency, self.T, self.P, self.rho, self.w, self.t_cloud)

    class opacity:
        @staticmethod
        def tau_oxygen(frequency: float,
                       T: Tensor1D_or_3D,
                       P: Tensor1D_or_3D,
                       dh: Union[float, Tensor1D]) -> Union[float, Tensor2D]:
            """
            :return: полное поглощение в кислороде (путем интегрирования). В неперах.
            """
            gamma = Model.attenuation.gamma_oxygen(frequency, T, P)
            return dB2np * Model._compat.integrate(gamma, dh)

        @staticmethod
        def tau_water_vapor(frequency: float,
                            T: Tensor1D_or_3D,
                            P: Tensor1D_or_3D,
                            rho: Tensor1D_or_3D,
                            dh: Union[float, Tensor1D]) -> Union[float, Tensor2D]:
            """
            :return: полное поглощение в водяном паре (путем интегрирования). В неперах.
            """
            gamma = Model.attenuation.gamma_water_vapor(frequency, T, P, rho)
            return dB2np * Model._compat.integrate(gamma, dh)

        @staticmethod
        def tau_liquid_water(frequency: float, t_cloud: float,
                             w: Union[float, Tensor3D], dh: Union[float, Tensor1D]) -> Union[float, Tensor2D]:
            gamma = Model.attenuation.gamma_liquid_water(frequency, t_cloud, w)
            return dB2np * Model._compat.integrate(gamma, dh)

    @storage
    def tau_oxygen(self, frequency: float) -> Union[float, Tensor2D]:
        return Model.opacity.tau_oxygen(frequency, self.T, self.P, self.dh)

    @storage
    def tau_water_vapor(self, frequency: float) -> Union[float, Tensor2D]:
        return Model.opacity.tau_water_vapor(frequency, self.T, self.P, self.rho, self.dh)

    @storage
    def tau_liquid_water(self, frequency: float) -> Union[float, Tensor2D]:
        return Model.opacity.tau_liquid_water(frequency, self.t_cloud, self.w, self.dh)

    @storage
    def tau_summary(self, frequency: float) -> Union[float, Tensor2D]:
        return self.tau_oxygen(frequency) + self.tau_water_vapor(frequency) + self.tau_liquid_water(frequency)

    class reflection:
        class smooth_water:
            @staticmethod
            def M_horizontal(frequency: float, psi: float, T: Union[float, Tensor2D],
                             Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
                epsilon = Model.dielectric.epsilon_complex(frequency, T, Sw)
                cos = Model._compat.sqrt(epsilon - np.cos(psi) * np.cos(psi))
                return (np.sin(psi) - cos) / (np.sin(psi) + cos)

            @staticmethod
            def M_vertical(frequency: float, psi: float, T: Union[float, Tensor2D],
                           Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
                epsilon = Model.dielectric.epsilon_complex(frequency, T, Sw)
                cos = Model._compat.sqrt(epsilon - np.cos(psi) * np.cos(psi))
                return (epsilon * np.sin(psi) - cos) / (epsilon * np.sin(psi) + cos)

            @staticmethod
            def R_horizontal(frequency: float, theta: float, T: Union[float, Tensor2D],
                             Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
                M_h = Model.reflection.smooth_water.M_horizontal(frequency, np.pi / 2. - theta, T, Sw)
                val = Model._compat.abs(M_h)
                return val * val

            @staticmethod
            def R_vertical(frequency: float, theta: float, T: Union[float, Tensor2D],
                           Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
                M_v = Model.reflection.smooth_water.M_vertical(frequency, np.pi / 2. - theta, T, Sw)
                val = Model._compat.abs(M_v)
                return val * val

            @staticmethod
            def R(frequency: float, T: Union[float, Tensor2D],
                  Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
                epsilon = Model.dielectric.epsilon_complex(frequency, T, Sw)
                val = Model._compat.abs((Model._compat.sqrt(epsilon) - 1) / (Model._compat.sqrt(epsilon) + 1))
                return val * val

    @storage
    def smooth_water_reflectance(self, frequency: float) -> Union[float, Tensor2D]:
        return Model.reflection.smooth_water.R(frequency, self.Tsurface_ocean, self.Salinity_ocean)

    class Downward:
        @staticmethod
        def brightness_temperature(frequency: float,
                                   T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                   rho: Tensor1D_or_3D, w: Tensor3D,
                                   t_cloud: float,
                                   dh: Union[float, Tensor1D],
                                   TK: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
            g = Model.attenuation.gamma_summary(frequency, T, P, rho, w, t_cloud, nepers=True)
            f = lambda h: (Model._compat.at(T, h) + 273.15) * Model._compat.at(g, h) * \
                Model._compat.exp(-Model._compat.integrate_bounds(g, 0, h, dh))
            inf = Model._compat.last_index(g)
            return Model._compat.integrate_callable(f, 0, inf, dh) + \
                TK * Model._compat.exp(-Model._compat.integrate(g, dh))

        @staticmethod
        def brightness_temperatures(
                frequencies: Union[np.ndarray, List[float], Iterable[float]],
                T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                rho: Tensor1D_or_3D, w: Tensor3D,
                t_cloud: float, dh: Union[float, Tensor1D], TK: Union[float, Tensor2D] = 0.,
                n_workers: int = None) -> np.ndarray:
            if not gpu:
                return Model._multi.parallel(frequencies,
                                             func=Model.Downward.brightness_temperature,
                                             args=(T, P, rho, w, t_cloud, dh, TK,),
                                             n_workers=n_workers)
            return np.asarray([Model.Downward.brightness_temperature(f, T, P, rho, w, t_cloud, dh, TK)
                               for f in frequencies], dtype=object)

        @staticmethod
        def T_avg(frequency: float,
                  T_standard: Tensor1D, P_standard: Tensor1D, rho_standard: Tensor1D,
                  dh: Union[float, Tensor1D], TK: Union[float, Tensor2D] = 0.) -> Union[float, TensorLike]:
            g = Model.attenuation.gamma_summary(frequency, T_standard, P_standard, rho_standard, nepers=True)
            f = lambda h: (Model._compat.at(T_standard, h) + 273.15) * Model._compat.at(g, h) * \
                Model._compat.exp(-Model._compat.integrate_bounds(g, 0, h, dh))
            inf = Model._compat.last_index(g)
            TauExp = Model._compat.exp(-Model._compat.integrate(g, dh))
            Tb_down = Model._compat.integrate_callable(f, 0, inf, dh) + TK * TauExp
            return Tb_down / (1. - TauExp)

    @storage
    def brightness_temperature_downward(self, frequency: float) -> Union[float, Tensor2D]:
        g = dB2np * self.gamma_summary(frequency)
        f = lambda h: (Model._compat.at(self.T, h) + 273.15) * Model._compat.at(g, h) * \
            Model._compat.exp(-Model._compat.integrate_bounds(g, 0, h, self.dh))
        inf = Model._compat.last_index(g)
        TauExp = Model._compat.exp(-self.tau_summary(frequency))
        return Model._compat.integrate_callable(f, 0, inf, self.dh) + self.T_cosmic * TauExp

    def brightness_temperatures_downward(self, frequencies: Union[np.ndarray, List[float], Iterable[float]],
                                         n_workers: int = None):
        if not gpu:
            return Model._multi.parallel(frequencies,
                                         func=self.brightness_temperature_downward,
                                         args=(), n_workers=n_workers)
        return np.asarray([self.brightness_temperature_downward(f) for f in frequencies], dtype=object)

    @storage
    def T_avg_downward(self, frequency: float) -> Union[float, TensorLike]:
        return Model.Downward.T_avg(frequency, self.T_std, self.P_std, self.rho_std, self.dh, self.T_cosmic)

    class Upward:
        @staticmethod
        def brightness_temperature(frequency: float,
                                   T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                   rho: Tensor1D_or_3D, w: Tensor3D,
                                   t_cloud: float, dh: Union[float, Tensor1D]) -> Union[float, Tensor2D]:
            g = Model.attenuation.gamma_summary(frequency, T, P, rho, w, t_cloud, nepers=True)
            inf = Model._compat.last_index(g)
            f = lambda h: (Model._compat.at(T, h) + 273.15) * Model._compat.at(g, h) * \
                Model._compat.exp(-Model._compat.integrate_bounds(g, h, inf, dh))
            return Model._compat.integrate_callable(f, 0, inf, dh)

        @staticmethod
        def brightness_temperatures(
                frequencies: Union[np.ndarray, List[float], Iterable[float]],
                T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                rho: Tensor1D_or_3D, w: Tensor3D,
                t_cloud: float, dh: Union[float, Tensor1D], n_workers: int = None) -> np.ndarray:
            if not gpu:
                return Model._multi.parallel(frequencies,
                                             func=Model.Upward.brightness_temperature,
                                             args=(T, P, rho, w, t_cloud, dh,),
                                             n_workers=n_workers)
            return np.asarray([Model.Upward.brightness_temperature(f, T, P, rho, w, t_cloud, dh, )
                               for f in frequencies], dtype=object)

        @staticmethod
        def T_avg(frequency: float,
                  T_standard: Tensor1D, P_standard: Tensor1D, rho_standard: Tensor1D,
                  dh: Union[float, Tensor1D]) -> Union[float, TensorLike]:
            g = Model.attenuation.gamma_summary(frequency, T_standard, P_standard, rho_standard, nepers=True)
            inf = Model._compat.last_index(g)
            f = lambda h: (Model._compat.at(T_standard, h) + 273.15) * Model._compat.at(g, h) * \
                Model._compat.exp(-Model._compat.integrate_bounds(g, h, inf, dh))
            Tb_up = Model._compat.integrate_callable(f, 0, inf, dh)
            TauExp = Model._compat.exp(-Model._compat.integrate(g, dh))
            return Tb_up / (1. - TauExp)

    @storage
    def brightness_temperature_upward(self, frequency: float) -> Union[float, Tensor2D]:
        g = dB2np * self.gamma_summary(frequency)
        inf = Model._compat.last_index(g)
        f = lambda h: (Model._compat.at(self.T, h) + 273.15) * Model._compat.at(g, h) * \
            Model._compat.exp(-Model._compat.integrate_bounds(g, h, inf, dh))
        return Model._compat.integrate_callable(f, 0, inf, dh)

    def brightness_temperatures_upward(self, frequencies: Union[np.ndarray, List[float], Iterable[float]],
                                       n_workers: int = None):
        if not gpu:
            return Model._multi.parallel(frequencies,
                                         func=self.brightness_temperature_upward,
                                         args=(), n_workers=n_workers)
        return np.asarray([self.brightness_temperature_upward(f) for f in frequencies], dtype=object)

    @storage
    def T_avg_upward(self, frequency: float) -> Union[float, TensorLike]:
        return Model.Upward.T_avg(frequency, self.T_std, self.P_std, self.rho_std, self.dh)

    class Satellite:
        @staticmethod
        def brightness_temperature(frequency: float,
                                   Tsurface_ocean: Union[float, Tensor2D],
                                   Salinity_ocean: Union[float, Tensor2D],
                                   T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                   rho: Tensor1D_or_3D, w: Tensor3D,
                                   t_cloud: float, dh: float,
                                   TK: float = 0.) -> Union[float, Tensor2D]:
            g = Model.attenuation.gamma_summary(frequency, T, P, rho, w, t_cloud, nepers=True)
            inf = Model._compat.last_index(g)

            f_down = lambda h: (Model._compat.at(T, h) + 273.15) * Model._compat.at(g, h) * \
                Model._compat.exp(-Model._compat.integrate_bounds(g, 0, h, dh))
            f_up = lambda h: (Model._compat.at(T, h) + 273.15) * Model._compat.at(g, h) * \
                Model._compat.exp(-Model._compat.integrate_bounds(g, h, inf, dh))

            TauExp = Model._compat.exp(-Model._compat.integrate(g, dh))
            Tb_down: Union[float, np.ndarray, tf.Tensor] = \
                Model._compat.integrate_callable(f_down, 0, inf, dh) + TK * TauExp
            Tb_up: Union[float, np.ndarray, tf.Tensor] = \
                Model._compat.integrate_callable(f_up, 0, inf, dh)

            R = Model.reflection.smooth_water.R(frequency, Tsurface_ocean, Salinity_ocean)
            kappa = 1. - R
            if not gpu:
                return (Tsurface_ocean + 273.15) * kappa * TauExp + Tb_up + R * Tb_down * TauExp
            return Model._compat.to_tensor((Tsurface_ocean + 273.15) * kappa) * TauExp + Tb_up + \
                Model._compat.to_tensor(R) * Tb_down * TauExp

        @staticmethod
        def brightness_temperatures(frequencies: Union[np.ndarray, List[float], Iterable[float]],
                                    Tsurface_ocean: Union[float, Tensor2D],
                                    Salinity_ocean: Union[float, Tensor2D],
                                    T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                    rho: Tensor1D_or_3D, w: Tensor3D,
                                    t_cloud: float, dh: float,
                                    TK: float = 0., n_workers: int = None) -> np.ndarray:
            if not gpu:
                return Model._multi.parallel(frequencies,
                                             func=Model.Satellite.brightness_temperature,
                                             args=(Tsurface_ocean, Salinity_ocean, T, P, rho, w, t_cloud, dh, TK,),
                                             n_workers=n_workers)
            return np.asarray([Model.Satellite.brightness_temperature(
                f, Tsurface_ocean, Salinity_ocean, T, P, rho, w, t_cloud, dh, TK, )
                for f in frequencies], dtype=object)

    @storage
    def brightness_temperature_satellite(self, frequency: float):
        TauExp = Model._compat.exp(-self.tau_summary(frequency))
        Tb_down = self.brightness_temperature_downward(frequency)
        Tb_up = self.brightness_temperature_upward(frequency)
        R = self.smooth_water_reflectance(frequency)
        kappa = 1. - R
        if not gpu:
            return (self.Tsurface_ocean + 273.15) * kappa * TauExp + Tb_up + R * Tb_down * TauExp
        return Model._compat.to_tensor((self.Tsurface_ocean + 273.15) * kappa) * TauExp + Tb_up + \
            Model._compat.to_tensor(R) * Tb_down * TauExp

    def brightness_temperatures_satellite(self, frequencies: Union[np.ndarray, List[float], Iterable[float]],
                                          n_workers: int = None):
        if not gpu:
            def __(frequency: float):
                key = (frequency, self.brightness_temperature_satellite.__name__)
                if key not in self._storage:
                    self._storage[key] = Model.brightness_temperature_satellite(self, frequency)
                return self._storage[key]
            return Model._multi.parallel(frequencies,
                                         func=__,
                                         args=(), n_workers=n_workers)
        return np.asarray([self.brightness_temperature_satellite(f) for f in frequencies], dtype=object)

    class Inverse:
        class downward:
            @staticmethod
            def opacity(tb: Union[float, Tensor2D], t_avg: Union[float, TensorLike],
                        theta: float = None) -> Union[float, Tensor2D]:
                if theta is None:
                    return Model._compat.log(t_avg) - Model._compat.log(t_avg - tb)
                return (Model._compat.log(t_avg) - Model._compat.log(t_avg - tb)) * np.cos(theta)

        # class satellite:
        #     @staticmethod
        #     def opacity(frequency: float,
        #                 tb: Union[float, Tensor2D],
        #                 theta: float = None, polarization: str = None,
        #                 T_standard: Tensor1D, P_standard: Tensor1D, rho_standard: Tensor1D,
        #                 Tsurface: Union[float, Tensor2D] = 15.,
        #                 Salinity: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    nx, ny = 300, 300
    # nx, ny = None, None
    H = 10
    N = 500
    dh = H / (N - 1.)

    # sp = Standard(H=H, dh=dh, nx=nx, ny=ny, near_ground=False)
    # T, P, rho = sp.T(celsius=True), sp.P(), sp.rho()
    #
    # domain = Planck((50, 50, 10), (nx, ny, N))
    # hmap = domain.create_height_map(
    #     Dm=3., K=100, alpha=1., beta=0.5, eta=1., seed=42
    # )
    # # plt.figure('height map')
    # # plt.imshow(hmap)
    # # plt.colorbar()
    # # plt.show()
    # # hmap = np.zeros((1, 1, 500))
    # w = domain.liquid_water_distribution(height_map=hmap, const_w=False)
    #
    # # freqs, tbs = [], []
    # # with open('tbs_check.txt', 'r') as file:
    # #     for line in file:
    # #         line = re.split(r'[ \t]', re.sub(r'[\r\n]', '', line))
    # #         f, tb = [float(n) for n in line if n]
    # #         freqs.append(f)
    # #         tbs.append(tb)
    #
    # start_time = time.time()
    # brt = Model.Satellite.brightness_temperatures([18.0, 22.2, 27.2], 15., 0., T, P, rho, w, -2., dh, n_workers=8)
    # # brt = Model.Satellite.Multifreq.brightness_temperature(freqs, 15., 0., T, P, rho, w, -2., dh, n_workers=8)
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    # plt.figure('brightness temperature')
    # # plt.ylim((50, 300))
    # # plt.scatter(freqs, tbs, label='test', marker='x', color='black')
    # plt.imshow(Model._compat.to_numpy_array(brt[1], float))
    # # plt.plot(freqs, [Model.compat.to_numpy_array(brt[j], float)[0][0] for j in range(len(freqs))], label='result')
    # plt.colorbar()
    # # plt.legend(loc='best', frameon=False)
    # plt.savefig('tbs_check.png', dpi=300)
    # plt.show()

    std = Standard(H=H, dh=dh, nx=nx, ny=ny, near_ground=False)
    model = Model(std.T(celsius=True), std.P(), std.rho(), dh=dh)
    domain = Planck((50, 50, 10), (nx, ny, N))
    hmap = domain.create_height_map(
        Dm=3., K=100, alpha=1., beta=0.5, eta=1., seed=42
    )
    model.set(T=std.T(celsius=True), P=std.P(), rho=std.rho())
    model.set(w=domain.liquid_water_distribution(hmap), t_cloud=-2.)
    model.set(Tsurface_ocean=15., Salinity_ocean=0.)

    start_time = time.time()
    # brt = model.Satellite(model).brightness_temperatures([18.0, 22.2, 27.2], n_workers=8)
    # tau = model.tau_summary(22.2)
    brt = model.brightness_temperatures_satellite([18.0])
    print("--- %s seconds ---" % (time.time() - start_time))
    print(model._storage)

    start_time = time.time()
    brt = model.brightness_temperatures_satellite([18.0, 22.2, 27.2, 37.5])
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure('brightness temperature')
    plt.imshow(Model._compat.to_numpy_array(brt[1], float))
    # plt.imshow(Model._compat.to_numpy_array(tau, float))
    # plt.imshow(Model._compat.to_numpy_array(brt, float))
    plt.colorbar()
    plt.savefig('tbs_check.png', dpi=300)
    plt.show()
