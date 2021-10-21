# -*- coding: utf-8 -*-

from typing import Union, List, Iterable, Tuple, Callable, Any
from multiprocessing import Manager, Process
from scipy.special import gamma
import numpy as np
import time
# import re

from cloudforms import CylinderCloud
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
Tensor1D, Tensor2D, Tensor3D, Tensor1D_or_3D = TensorLike, TensorLike, TensorLike, TensorLike


class StandardProfiles:
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
    def _altitudes(self) -> np.ndarray:
        if self.near_ground:
            return np.asarray([0])
        return np.arange(0, self.H + self.dh / 2., self.dh)

    def altitudes(self) -> Tensor1D:
        return self.gpu(self._altitudes)

    @staticmethod
    def reshape(profile: np.ndarray, sizes: tuple) -> np.ndarray:
        nx, ny = sizes
        if nx and ny:
            profile = np.asarray([[profile for _ in range(ny)] for _ in range(nx)], dtype=cpu_float)
        return profile

    @staticmethod
    def gpu(profile: np.ndarray) -> Tensor1D_or_3D:
        if gpu:
            return tf.convert_to_tensor(profile, dtype=gpu_float)
        return profile

    @staticmethod
    def __(profile: list) -> np.ndarray:
        if len(profile) == 1:
            profile = profile[0]
        return np.asarray(profile, dtype=cpu_float)

    def zeros(self) -> Tensor1D_or_3D:
        return self.gpu(self.reshape(np.zeros(len(self._altitudes), dtype=cpu_float), (self.nx, self.ny)))

    def T(self, celsius=True) -> Tensor1D_or_3D:
        """
        :param celsius: град. Цельс.?
        :return: Стандартный высотный профиль термодинамической температуры
        (К или Цельс. в зависимости от celsius).
        """
        profile = []
        T11 = self.T0 - 6.5 * 11
        T32, T47 = 0., 0.
        for h in self._altitudes:
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

    def P(self, HP: float = 7.7) -> Tensor1D_or_3D:
        """
        :return: Стандартный высотный профиль атмосферного давления (гПа).
        """
        profile = self.__([self.P0 * np.exp(-h / HP) for h in self._altitudes])
        return self.gpu(self.reshape(profile, (self.nx, self.ny)))

    def rho(self, Hrho: float = 2.1) -> Tensor1D_or_3D:
        """
        :return: Стандартный высотный профиль абсолютной влажности (г/м^3).
        """
        profile = self.__([self.rho0 * np.exp(-h / Hrho) for h in self._altitudes])
        return self.gpu(self.reshape(profile, (self.nx, self.ny)))


class Domain:
    def __init__(self, kilometers, nodes):
        self.PX, self.PY, self.PZ = kilometers
        if nodes is None:
            nodes = (1, 1, 500)
        self.Nx, self.Ny, self.Nz = nodes
        if self.Nx is None:
            self.Nx = 1
        if self.Ny is None:
            self.Ny = 1
        if self.Nz is None:
            self.Nz = 500
        self.cl_bottom = 1.5

    def x(self, i: int) -> float:
        return i * self.PX / (self.Nx - 1)

    def y(self, j: int) -> float:
        return j * self.PY / (self.Ny - 1)

    def z(self, k: int) -> float:
        return k * self.PZ / (self.Nz - 1)

    def i(self, x: float) -> int:
        return int(x / self.PX * (self.Nx - 1))

    def j(self, y: float) -> int:
        return int(y / self.PY * (self.Ny - 1))

    def k(self, z: float) -> int:
        return int(z / self.PZ * (self.Nz - 1))

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

    class compat:

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
            return Model.compat.len(a) - 1

        @staticmethod
        def integrate(a: Tensor1D_or_3D,
                      dh: float, method='trapz') -> Union[Number, Tensor2D]:
            return Model.compat.integrate_bounds(a, 0, Model.compat.last_index(a), dh, method)

        @staticmethod
        def integrate_bounds(a: Tensor1D_or_3D, lower: int, upper: int,
                             dh: float, method='trapz') -> Union[Number, Tensor2D]:
            if method not in ['trapz', 'simpson', 'boole']:
                raise ValueError('выберите один из доступных методов: \'trapz\', \'simpson\', \'boole\'')
            rank = Model.compat.rank(a)
            if rank not in [1, 3]:
                raise RuntimeError('неверная размерность')
            if rank == 3:
                a = Model.compat.transpose(a, [2, 0, 1])
            if method == 'trapz':
                a = Model.compat.sum(a[lower+1:upper], axis=0) * dh + (a[lower] + a[upper]) / 2. * dh
            if method == 'simpson':
                a = (a[lower] + a[upper] +
                     4 * Model.compat.sum(a[lower+1:upper:2], axis=0) +
                     2 * Model.compat.sum(a[lower+2:upper:2], axis=0)) * dh / 3.
            if method == 'boole':
                a = (14 * (a[lower] + a[upper]) +
                     64 * Model.compat.sum(a[lower+1:upper:2], axis=0) +
                     24 * Model.compat.sum(a[lower+2:upper:4], axis=0) +
                     28 * Model.compat.sum(a[lower+4:upper:4], axis=0)) * dh / 45.
            return a

        @staticmethod
        def integrate_callable(f: Callable, lower: int, upper: int,
                               dh: float) -> Union[Number, TensorLike]:
            a = dh * (f(lower) + f(upper)) / 2.
            for k in range(lower + 1, upper):
                a += dh * f(k)
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
            rank = Model.compat.rank(a)
            if rank not in [1, 3]:
                raise RuntimeError('неверная размерность')
            if rank == 3:
                return a[:, :, index]
            return a[index]

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
                Model.multi.do(processes, n_workers)
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

    class integral:
        @staticmethod
        def tau_oxygen(frequency: float,
                       T: Tensor1D_or_3D,
                       P: Tensor1D_or_3D,
                       dh: float) -> Union[float, Tensor1D_or_3D]:
            """
            :return: полное поглощение в кислороде (путем интегрирования). В неперах.
            """
            gamma = Model.p676.gamma_oxygen(frequency, T, P)
            return dB2np * Model.compat.integrate(gamma, dh)

        @staticmethod
        def tau_water_vapor(frequency: float,
                            T: Tensor1D_or_3D,
                            P: Tensor1D_or_3D,
                            rho: Tensor1D_or_3D,
                            dh: float) -> Union[float, Tensor1D_or_3D]:
            """
            :return: полное поглощение в водяном паре (путем интегрирования). В неперах.
            """
            gamma = Model.p676.gamma_water_vapor(frequency, T, P, rho)
            return dB2np * Model.compat.integrate(gamma, dh)

    class dielectric:
        @staticmethod
        def epsilon(T: Union[float, Tensor2D],
                    Sw: Union[float, Tensor2D] = 0.) -> Tuple[float, float, float]:
            epsO_nosalt = 5.5
            epsS_nosalt = 88.2 - 0.40885 * T + 0.00081 * T * T
            lambdaS_nosalt = 1.8735116 - 0.027296 * T + 0.000136 * T * T + \
                1.662 * Model.compat.exp(-0.0634 * T)
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

    class wf:
        @staticmethod
        def krho(frequency: float,
                 T_standard: Tensor1D,
                 P_standard: Tensor1D,
                 rho_standard: Tensor1D, dh: float) -> float:
            return Model.integral.tau_water_vapor(frequency, T_standard, P_standard, rho_standard, dh) / \
                   (Model.compat.integrate(rho_standard, dh) / 10.)

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
                return np2dB * Model.wf.kw(frequency, t_cloud) * w
            return Model.compat.to_tensor(np2dB * Model.wf.kw(frequency, t_cloud)) * w

        @staticmethod
        def gamma_summary(frequency: float,
                          T: Union[float, Tensor1D_or_3D], P: Union[float, Tensor1D_or_3D],
                          rho: Union[float, Tensor1D_or_3D], w: Union[float, Tensor3D] = None,
                          t_cloud: float = None) -> Union[float, Tensor3D]:
            if w is None or t_cloud is None:
                return dB2np * (Model.attenuation.gamma_oxygen(frequency, T, P) +
                                Model.attenuation.gamma_water_vapor(frequency, T, P, rho))
            return dB2np * (Model.attenuation.gamma_oxygen(frequency, T, P) +
                            Model.attenuation.gamma_water_vapor(frequency, T, P, rho) +
                            Model.attenuation.gamma_liquid_water(frequency, t_cloud, w))

    class reflection:
        @staticmethod
        def M_horizontal(frequency: float, psi: float, T: Union[float, Tensor2D],
                         Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
            epsilon = Model.dielectric.epsilon_complex(frequency, T, Sw)
            cos = Model.compat.sqrt(epsilon - np.cos(psi) * np.cos(psi))
            return (np.sin(psi) - cos) / (np.sin(psi) + cos)

        @staticmethod
        def M_vertical(frequency: float, psi: float, T: Union[float, Tensor2D],
                       Sw: Union[float, Tensor2D] = 0.) -> Union[Number, Tensor2D]:
            epsilon = Model.dielectric.epsilon_complex(frequency, T, Sw)
            cos = Model.compat.sqrt(epsilon - np.cos(psi) * np.cos(psi))
            return (epsilon * np.sin(psi) - cos) / (epsilon * np.sin(psi) + cos)

        @staticmethod
        def R_horizontal(frequency: float, theta: float, T: Union[float, Tensor2D],
                         Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
            M_h = Model.reflection.M_horizontal(frequency, np.pi / 2. - theta, T, Sw)
            val = Model.compat.abs(M_h)
            return val * val

        @staticmethod
        def R_vertical(frequency: float, theta: float, T: Union[float, Tensor2D],
                       Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
            M_v = Model.reflection.M_vertical(frequency, np.pi / 2. - theta, T, Sw)
            val = Model.compat.abs(M_v)
            return val * val

        @staticmethod
        def R(frequency: float, T: Union[float, Tensor2D],
              Sw: Union[float, Tensor2D] = 0.) -> Union[float, Tensor2D]:
            epsilon = Model.dielectric.epsilon_complex(frequency, T, Sw)
            val = Model.compat.abs((Model.compat.sqrt(epsilon) - 1) / (Model.compat.sqrt(epsilon) + 1))
            return val * val

    class Downward:
        @staticmethod
        def brightness_temperature(frequency: float,
                                   T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                   rho: Tensor1D_or_3D, w: Tensor3D,
                                   t_cloud: float, dh: float, TK: float = 0.) -> Union[float, Tensor2D]:
            g = Model.attenuation.gamma_summary(frequency, T, P, rho, w, t_cloud)
            f = lambda h: (Model.compat.at(T, h) + 273.15) * Model.compat.at(g, h) * \
                Model.compat.exp(-Model.compat.integrate_bounds(g, 0, h, dh))
            inf = Model.compat.last_index(g)
            return Model.compat.integrate_callable(f, 0, inf, dh) + \
                TK * Model.compat.exp(-Model.compat.integrate(g, dh))

        @staticmethod
        def T_avg(frequency: float,
                  T_standard: Tensor1D, P_standard: Tensor1D, rho_standard: Tensor1D,
                  dh: float, TK: float = 0.) -> Union[float, TensorLike]:
            g = Model.attenuation.gamma_summary(frequency, T_standard, P_standard, rho_standard)
            f = lambda h: (Model.compat.at(T_standard, h) + 273.15) * Model.compat.at(g, h) * \
                Model.compat.exp(-Model.compat.integrate_bounds(g, 0, h, dh))
            inf = Model.compat.last_index(g)
            TauExp = Model.compat.exp(-Model.compat.integrate(g, dh))
            Tb_down = Model.compat.integrate_callable(f, 0, inf, dh) + TK * TauExp
            return Tb_down / (1. - TauExp)

        class multifreq:
            @staticmethod
            def brightness_temperature(
                    frequencies: Union[np.ndarray, List[float], Iterable[float]],
                    T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                    rho: Tensor1D_or_3D, w: Tensor3D,
                    t_cloud: float, dh: float, TK: float = 0.,
                    n_workers: int = None) -> np.ndarray:
                if Model.compat.are_numpy_arrays(T, P, rho, w):
                    return Model.multi.parallel(frequencies,
                                                func=Model.Downward.brightness_temperature,
                                                args=(T, P, rho, w, t_cloud, dh, TK,),
                                                n_workers=n_workers)
                elif Model.compat.are_tensors(T, P, rho, w):
                    return np.asarray([Model.Downward.brightness_temperature(f, T, P, rho, w, t_cloud, dh, TK)
                                       for f in frequencies], dtype=object)
                raise TypeError('массивы должны быть одинаковых типов')

    class Upward:
        @staticmethod
        def brightness_temperature(frequency: float,
                                   T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                   rho: Tensor1D_or_3D, w: Tensor3D,
                                   t_cloud: float, dh: float) -> Union[float, Tensor2D]:
            g = Model.attenuation.gamma_summary(frequency, T, P, rho, w, t_cloud)
            inf = Model.compat.last_index(g)
            f = lambda h: (Model.compat.at(T, h) + 273.15) * Model.compat.at(g, h) * \
                Model.compat.exp(-Model.compat.integrate_bounds(g, h, inf, dh))
            return Model.compat.integrate_callable(f, 0, inf, dh)

        @staticmethod
        def T_avg(frequency: float,
                  T_standard: Tensor1D, P_standard: Tensor1D, rho_standard: Tensor1D,
                  dh: float) -> Union[float, TensorLike]:
            g = Model.attenuation.gamma_summary(frequency, T_standard, P_standard, rho_standard)
            inf = Model.compat.last_index(g)
            f = lambda h: (Model.compat.at(T_standard, h) + 273.15) * Model.compat.at(g, h) * \
                Model.compat.exp(-Model.compat.integrate_bounds(g, h, inf, dh))
            Tb_up = Model.compat.integrate_callable(f, 0, inf, dh)
            TauExp = Model.compat.exp(-Model.compat.integrate(g, dh))
            return Tb_up / (1. - TauExp)

        class multifreq:
            @staticmethod
            def brightness_temperature(
                    frequencies: Union[np.ndarray, List[float], Iterable[float]],
                    T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                    rho: Tensor1D_or_3D, w: Tensor3D,
                    t_cloud: float, dh: float, n_workers: int = None) -> np.ndarray:
                if Model.compat.are_numpy_arrays(T, P, rho, w):
                    return Model.multi.parallel(frequencies,
                                                func=Model.Upward.brightness_temperature,
                                                args=(T, P, rho, w, t_cloud, dh,),
                                                n_workers=n_workers)
                elif Model.compat.are_tensors(T, P, rho, w):
                    return np.asarray([Model.Upward.brightness_temperature(f, T, P, rho, w, t_cloud, dh,)
                                       for f in frequencies], dtype=object)
                raise TypeError('массивы должны быть одинаковых типов')

    class Satellite:
        @staticmethod
        def brightness_temperature(frequency: float,
                                   Tsurface: Union[float, Tensor2D],
                                   Salinity: Union[float, Tensor2D],
                                   T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                   rho: Tensor1D_or_3D, w: Tensor3D,
                                   t_cloud: float, dh: float,
                                   TK: float = 0.) -> Union[float, Tensor2D]:
            g = Model.attenuation.gamma_summary(frequency, T, P, rho, w, t_cloud)
            inf = Model.compat.last_index(g)

            f_down = lambda h: (Model.compat.at(T, h) + 273.15) * Model.compat.at(g, h) * \
                Model.compat.exp(-Model.compat.integrate_bounds(g, 0, h, dh))
            f_up = lambda h: (Model.compat.at(T, h) + 273.15) * Model.compat.at(g, h) * \
                Model.compat.exp(-Model.compat.integrate_bounds(g, h, inf, dh))

            TauExp = Model.compat.exp(-Model.compat.integrate(g, dh))
            Tb_down: Union[float, np.ndarray, tf.Tensor] = \
                Model.compat.integrate_callable(f_down, 0, inf, dh) + TK * TauExp
            Tb_up: Union[float, np.ndarray, tf.Tensor] = \
                Model.compat.integrate_callable(f_up, 0, inf, dh)

            R = Model.reflection.R(frequency, Tsurface, Salinity)
            kappa = 1. - R
            if isinstance(g, (float, np.ndarray)):
                return (Tsurface + 273.15) * kappa * TauExp + Tb_up + R * Tb_down * TauExp
            return Model.compat.to_tensor((Tsurface + 273.15) * kappa) * TauExp + Tb_up + \
                Model.compat.to_tensor(R) * Tb_down * TauExp

        class multifreq:
            @staticmethod
            def brightness_temperature(frequencies: Union[np.ndarray, List[float], Iterable[float]],
                                       Tsurface: Union[float, Tensor2D],
                                       Salinity: Union[float, Tensor2D],
                                       T: Tensor1D_or_3D, P: Tensor1D_or_3D,
                                       rho: Tensor1D_or_3D, w: Tensor3D,
                                       t_cloud: float, dh: float,
                                       TK: float = 0., n_workers: int = None) -> np.ndarray:
                if Model.compat.are_numpy_arrays(T, P, rho, w):
                    return Model.multi.parallel(frequencies,
                                                func=Model.Satellite.brightness_temperature,
                                                args=(Tsurface, Salinity, T, P, rho, w, t_cloud, dh, TK,),
                                                n_workers=n_workers)
                elif Model.compat.are_tensors(T, P, rho, w):
                    return np.asarray([Model.Satellite.brightness_temperature(
                        f, Tsurface, Salinity, T, P, rho, w, t_cloud, dh, TK,)
                                       for f in frequencies], dtype=object)
                raise TypeError('массивы должны быть одинаковых типов')

    class Inverse:
        class downward:
            @staticmethod
            def opacity(tb: Union[float, Tensor2D], t_avg: Union[float, TensorLike],
                        theta: float = None) -> Union[float, Tensor2D]:
                if theta is None:
                    return Model.compat.log(t_avg) - Model.compat.log(t_avg - tb)
                return (Model.compat.log(t_avg) - Model.compat.log(t_avg - tb)) * np.cos(theta)

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

    sp = StandardProfiles(H=H, dh=dh, nx=nx, ny=ny, near_ground=False)
    T, P, rho = sp.T(celsius=True), sp.P(), sp.rho()

    domain = Domain((50, 50, 10), (nx, ny, N))
    hmap = domain.create_height_map(
        Dm=3., K=100, alpha=1., beta=0.5, eta=1., seed=42
    )
    # plt.figure('height map')
    # plt.imshow(hmap)
    # plt.colorbar()
    # plt.show()
    # hmap = np.zeros((1, 1, 500))
    w = domain.liquid_water_distribution(height_map=hmap, const_w=False)

    # freqs, tbs = [], []
    # with open('tbs_check.txt', 'r') as file:
    #     for line in file:
    #         line = re.split(r'[ \t]', re.sub(r'[\r\n]', '', line))
    #         f, tb = [float(n) for n in line if n]
    #         freqs.append(f)
    #         tbs.append(tb)

    start_time = time.time()
    brt = Model.Satellite.multifreq.brightness_temperature([18.0, 22.2, 27.2], 15., 0., T, P, rho, w, -2., dh,
                                                           n_workers=8)
    # brt = Model.Satellite.Multifreq.brightness_temperature(freqs, 15., 0., T, P, rho, w, -2., dh, n_workers=8)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure('brightness temperature')
    # plt.ylim((50, 300))
    # plt.scatter(freqs, tbs, label='test', marker='x', color='black')
    plt.imshow(Model.compat.to_numpy_array(brt[1], float))
    # plt.plot(freqs, [Model.compat.to_numpy_array(brt[j], float)[0][0] for j in range(len(freqs))], label='result')
    plt.colorbar()
    # plt.legend(loc='best', frameon=False)
    plt.savefig('tbs_check.png', dpi=300)
    plt.show()
