# -*- coding: utf-8 -*-
from typing import Tuple
from domain import Domain3D
from cloudforms import CylinderCloud
import numpy as np
import time
from scipy.special import gamma


class Plank(Domain3D):
    def __init__(self, kilometers: Tuple[float, float, float] = (50., 50., 10.),
                 nodes: Tuple[int, int, int] = (300, 300, 500),
                 clouds_bottom: float = 1.5):
        """
        Модель Планка разрывной кучевой облачности в 3D

        :param kilometers: размеры по осям Ox, Oy и Oz в километрах
        :param nodes: кол-во узлов по соответствующим осям
        :param clouds_bottom: высота нижней границы облаков
        """
        super().__init__(kilometers, nodes)
        self.clouds_bottom = clouds_bottom

    @classmethod
    def from_domain3D(cls, domain: 'Domain3D', clouds_bottom: float = 1.5):
        return cls((domain.PX, domain.PY, domain.PZ),
                   (domain.Nx, domain.Ny, domain.Nz), clouds_bottom=clouds_bottom)

    def cloudiness(self, Dm: float = 3., K: float = 100,
                   alpha: float = 1., beta: float = 0.5, eta: float = 1., seed: int = 42,
                   timeout: float = 30.,
                   verbose=True) -> list:
        """
        :param Dm: максимальный диаметр облака, км
        :param K: нормировочный коэффициент, безразм.
        :param alpha: безразм. коэфф.
        :param beta: безразм. коэфф.
        :param eta: безразм. коэфф.
        :param seed: состояние генератора случайных чисел (определяет положения облаков в 3D)
        :param timeout: максимальное время ожидания
        :param verbose: вывод доп. информации
        :return: 2D-распределение мощности облаков в проекции на плоскость Oxy
        """
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
                    x, y = np.random.uniform(0., self.PX), np.random.uniform(0., self.PY)
                    z = self.clouds_bottom
                    rx = ry = D / 2
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
                    if time.time() - start_time > timeout:
                        raise TimeoutError('превышено допустимое время ожидания')
        if verbose:
            print()
        return cloudiness

    def h_map(self, Dm: float = 3., K: float = 100,
              alpha: float = 1., beta: float = 0.5, eta: float = 1., seed: int = 42,
              timeout: float = 30.,
              verbose=True) -> np.ndarray:
        """
        :param Dm: максимальный диаметр облака, км
        :param K: нормировочный коэффициент, безразм.
        :param alpha: безразм. коэфф.
        :param beta: безразм. коэфф.
        :param eta: безразм. коэфф.
        :param seed: состояние генератора случайных чисел (определяет положения облаков в 3D)
        :param timeout: максимальное время ожидания
        :param verbose: вывод доп. информации
        :return: 2D-распределение мощности облаков в проекции на плоскость Oxy
        """
        cloudiness = self.cloudiness(Dm, K, alpha, beta, eta, seed, timeout, verbose)
        hmap = np.zeros((self.Nx, self.Ny), dtype=float)
        for cloud in cloudiness:
            for x in np.arange(cloud.x - cloud.rx, cloud.x + cloud.rx, self.dx):
                for y in np.arange(cloud.y - cloud.ry, cloud.y + cloud.ry, self.dy):
                    if cloud.includesQ((x, y, self.clouds_bottom)):
                        hmap[self.i(x), self.j(y)] = cloud.height
        return hmap     # 2D array

    def lw_dist(self, height_map: np.ndarray, const_w=False,
                mu0: float = 3.27, psi0: float = 0.67) -> np.ndarray:
        """
        Расчет 3D поля водности по заданному 2D-распределению мощности облаков

        :param height_map: 2D-распределение мощности облаков в проекции на плоскость Oxy
        :param const_w: если True, внутри облака водность не меняется с высотой; если False, используется модель Мазина
        :param mu0: безразм. параметр
        :param psi0: безразм. параметр
        :return: поле водности в 3D
        """
        min_level = self.k(self.clouds_bottom)
        max_level = self.k(self.clouds_bottom + np.max(height_map))
        W = 0.132574 * np.power(height_map, 2.30215)
        w = np.zeros((self.Nx, self.Ny, self.Nz), dtype=float)
        cond = np.logical_not(np.isclose(height_map, 0.))

        if const_w:
            for k in range(min_level, max_level):
                xi = (self.z(k) - self.clouds_bottom) / height_map[cond]
                xi[(xi < 0) | (xi > 1)] = 0.
                xi[(0 <= xi) | (xi <= 1)] = 1.
                w[cond, k] = xi * W[cond] / height_map[cond]
        else:
            for k in range(min_level, max_level):
                xi = (self.z(k) - self.clouds_bottom) / height_map[cond]
                xi[(xi < 0) | (xi > 1)] = 0.
                w[cond, k] = \
                    np.power(xi, mu0) * np.power(1 - xi, psi0) * W[cond] / height_map[cond] * \
                    gamma(2 + mu0 + psi0) / (gamma(1 + mu0) * gamma(1 + psi0))

        return w    # 3D array

    def get_lw_dist(self, Dm: float = 3., K: float = 100,
                    alpha: float = 1., beta: float = 0.5, eta: float = 1., seed: int = 42,
                    verbose=True, const_w=False,
                    mu0: float = 3.27, psi0: float = 0.67) -> np.ndarray:
        """
        Выполняет lw_dist(h_map(...), ...)
        """
        return self.lw_dist(self.h_map(Dm, K, alpha, beta, eta, seed, verbose), const_w, mu0, psi0)     # 3D array
