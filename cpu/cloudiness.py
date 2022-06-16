# -*- coding: utf-8 -*-
from typing import Union, Callable
from cpu.core.domain import Domain3D, Column3D
from cpu.core.cloudforms import *
import numpy as np
from scipy.special import gamma
import time


def liquid_water(domain: Union[Domain3D, Column3D],
                 height_map2d: np.ndarray,
                 clouds_bottom: float = 1.5,
                 const_w: bool = False,
                 mu0: float = 3.27, psi0: float = 0.67,
                 _w: Callable = lambda _h: 0.132574 * np.power(_h, 2.30215)) -> np.ndarray:
    """
    Расчет 3D поля водности по заданному 2D-распределению мощности облаков

    :param domain: объект класса Домен - расчетная область
    :param height_map2d: 2D-распределение мощности облаков в проекции на плоскость Oxy
    :param clouds_bottom: высота нижней границы облаков
    :param const_w: если True, внутри облака водность не меняется с высотой; если False, используется модель Мазина
    :param mu0: безразмерный параметр
    :param psi0: безразмерный параметр
    :param _w: зависимость водозапаса от мощности облака
    :return: поле водности в 3D
    """
    min_level = domain.k(clouds_bottom)
    max_level = domain.k(clouds_bottom + np.max(height_map2d))
    # w_map2d = 0.132574 * np.power(height_map2d, 2.30215)
    w_map2d = _w(height_map2d)
    w = np.zeros(domain.nodes, dtype=float)
    cond = np.logical_not(np.isclose(height_map2d, 0.))

    if const_w:
        for k in range(min_level, max_level):
            xi = (domain.z(k) - clouds_bottom) / height_map2d[cond]
            xi[(xi < 0) | (xi > 1)] = 0.
            xi[(0 <= xi) | (xi <= 1)] = 1.
            w[cond, k] = xi * w_map2d[cond] / height_map2d[cond]
    else:
        for k in range(min_level, max_level):
            xi = (domain.z(k) - clouds_bottom) / height_map2d[cond]
            xi[(xi < 0) | (xi > 1)] = 0.
            w[cond, k] = \
                np.power(xi, mu0) * np.power(1 - xi, psi0) * w_map2d[cond] / height_map2d[cond] * \
                gamma(2 + mu0 + psi0) / (gamma(1 + mu0) * gamma(1 + psi0))
    return w  # 3D array


class Cloudiness3D(Domain3D):
    def __init__(self, kilometers: Tuple[float, float, float] = (50., 50., 10.),
                 nodes: Tuple[int, int, int] = (300, 300, 500), clouds_bottom: float = 1.5):
        super().__init__(kilometers, nodes)
        self.clouds_bottom = clouds_bottom

    @classmethod
    def from_domain(cls, domain: Domain3D, clouds_bottom: float = 1.5):
        return cls(domain.kilometers, domain.nodes, clouds_bottom)

    def liquid_water(self, height_map2d: np.ndarray, const_w: bool = False,
                     mu0: float = 3.27, psi0: float = 0.67,
                     _w: Callable = lambda _h: 0.132574 * np.power(_h, 2.30215)) -> np.ndarray:
        return liquid_water(self, height_map2d, self.clouds_bottom, const_w, mu0, psi0, _w)  # 3D array


class CloudinessColumn(Column3D):
    def __init__(self, kilometers_z: float = 10., nodes_z: int = 500, clouds_bottom: float = 1.5):
        super().__init__(kilometers_z, nodes_z)
        self.clouds_bottom = clouds_bottom

    @classmethod
    def from_domain(cls, column: Column3D, clouds_bottom: float = 1.5):
        return cls(column.PZ, column.Nz, clouds_bottom)

    def liquid_water(self, height: float, const_w: bool = False,
                     mu0: float = 3.27, psi0: float = 0.67,
                     _w: Callable = lambda _h: 0.132574 * np.power(_h, 2.30215)) -> np.ndarray:
        return liquid_water(self, np.array([[height]]), self.clouds_bottom, const_w, mu0, psi0, _w)  # 3D array


class Plank3D(Domain3D):
    def __init__(self, kilometers: Tuple[float, float, float] = (50., 50., 10.),
                 nodes: Tuple[int, int, int] = (300, 300, 500), clouds_bottom: float = 1.5):
        super().__init__(kilometers, nodes)
        self.clouds_bottom = clouds_bottom

    @classmethod
    def from_domain(cls, domain: Domain3D, clouds_bottom: float = 1.5):
        return cls(domain.kilometers, domain.nodes, clouds_bottom)

    def generate_clouds(self, Dm: float = 3., dm: float = 0., K: float = 100,
                        alpha: float = 1., beta: float = 0.5, eta: float = 1., seed: int = 42,
                        timeout: float = 30., verbose=True) -> list:
        """
        :param Dm: максимальный диаметр облака, км
        :param dm: минимально возможный диаметр облака в км
        :param K: нормировочный коэффициент, безразмерный
        :param alpha: безразмерный коэфф.
        :param beta: безразмерный коэфф.
        :param eta: безразмерный коэфф.
        :param seed: состояние генератора случайных чисел (определяет положения облаков в 3D)
        :param timeout: максимальное время ожидания
        :param verbose: вывод доп. информации
        :return: список облаков
        """
        np.random.seed(seed)
        clouds = []
        # вариант 1 - неправильно
        # r = np.sqrt(self.i(Dm) * self.i(Dm) + self.j(Dm) * self.j(Dm))
        # steps = np.arange(Dm, dm, -(Dm - dm) / r)
        # N = len(steps)
        # for i, D in enumerate(steps):
        #     if verbose:
        #         print('\r{:.2f}%'.format((i + 1) / N * 100), end='', flush=True)
        #     n = int(np.round(K * np.exp(-alpha * D)))
        #     if n < 1:
        #         n = 1
        #     for k in range(n):
        #         start_time = time.time()
        #         while True:
        #             x, y = np.random.uniform(0., self.PX), np.random.uniform(0., self.PY)
        #             z = self.clouds_bottom
        #             rx = ry = D / 2
        #             H = eta * D * np.power(D / Dm, beta)
        #             cloud = CylinderCloud((x, y, z), rx, ry, H)
        #             if not cloud.belongs_q((self.PX, self.PY, self.PZ)):
        #                 continue
        #             intersections = False
        #             for c in clouds:
        #                 if not cloud.disjoint_q(c):
        #                     intersections = True
        #                     break
        #             if not intersections:
        #                 clouds.append(cloud)
        #                 break
        #             if time.time() - start_time > timeout:
        #                 raise TimeoutError('timeout exceeded')

        # вариант 4
        r = np.sqrt(self.i(Dm) * self.i(Dm) + self.j(Dm) * self.j(Dm))
        eps = (Dm - dm) / r
        steps = np.arange(Dm, dm, -eps)
        N = len(steps)
        for i, D in enumerate(steps):
            if verbose:
                print('\r{:.2f}%'.format((i + 1) / N * 100), end='', flush=True)
            n = int(np.round(
                K / alpha * (np.exp(-alpha * D) - np.exp(-alpha * (D + eps)))
            ))
            if n < 1:
                print('\nw: отсутствуют облака диаметром {}'.format(D))
                continue
            for k in range(n):
                start_time = time.time()
                while True:
                    x, y = np.random.uniform(0., self.PX), np.random.uniform(0., self.PY)
                    z = self.clouds_bottom
                    rx = ry = D / 2
                    H = eta * D * np.power(D / Dm, beta)
                    cloud = CylinderCloud((x, y, z), rx, ry, H)
                    if not cloud.belongs_q((self.PX, self.PY, self.PZ)):
                        continue
                    intersections = False
                    for c in clouds:
                        if not cloud.disjoint_q(c):
                            intersections = True
                            break
                    if not intersections:
                        clouds.append(cloud)
                        break
                    if time.time() - start_time > timeout:
                        raise TimeoutError('timeout exceeded')
        if verbose:
            print()
        return clouds

    def height_map2d_(self, cloudiness: list) -> np.ndarray:
        hmap = np.zeros((self.Nx, self.Ny), dtype=float)
        for cloud in cloudiness:
            for x in np.arange(cloud.x - cloud.rx, cloud.x + cloud.rx, self.dx):
                for y in np.arange(cloud.y - cloud.ry, cloud.y + cloud.ry, self.dy):
                    if cloud.includes_q((x, y, self.clouds_bottom)):
                        hmap[self.i(x), self.j(y)] = cloud.height
        return hmap  # 2D array

    def height_map2d(self, Dm: float = 3., dm: float = 0., K: float = 100,
                     alpha: float = 1., beta: float = 0.5, eta: float = 1., seed: int = 42,
                     timeout: float = 30., verbose=True) -> np.ndarray:
        """
        :param Dm: максимальный диаметр облака, км
        :param dm: минимально возможный диаметр облака в км
        :param K: нормировочный коэффициент, безразмерный
        :param alpha: безразмерный коэфф.
        :param beta: безразмерный коэфф.
        :param eta: безразмерный коэфф.
        :param seed: состояние генератора случайных чисел (определяет положения облаков в 3D)
        :param timeout: максимальное время ожидания
        :param verbose: вывод доп. информации
        :return: 2D-распределение мощности облаков в проекции на плоскость Oxy
        """
        cloudiness = self.generate_clouds(Dm, dm, K, alpha, beta, eta, seed, timeout, verbose)
        return self.height_map2d_(cloudiness)

    def liquid_water_(self, hmap2d: np.ndarray, const_w=False, mu0: float = 3.27, psi0: float = 0.67,
                      _w: Callable = lambda _h: 0.132574 * np.power(_h, 2.30215)) -> np.ndarray:
        return liquid_water(self, hmap2d, self.clouds_bottom, const_w, mu0, psi0, _w)

    def liquid_water(self, Dm: float = 3., dm: float = 0., K: float = 100,
                     alpha: float = 1., beta: float = 0.5, eta: float = 1., seed: int = 42,
                     const_w=False, mu0: float = 3.27, psi0: float = 0.67,
                     _w: Callable = lambda _h: 0.132574 * np.power(_h, 2.30215),
                     timeout: float = 30., verbose=True) -> np.ndarray:
        return liquid_water(self, self.height_map2d(Dm, dm, K, alpha, beta, eta, seed, timeout, verbose),
                            self.clouds_bottom, const_w, mu0, psi0, _w)  # 3D array
