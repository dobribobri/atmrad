# -*- coding: utf-8 -*-
# import os
# import sys
# import warnings
import dill
# import datetime
import numpy as np
# from collections import defaultdict

from gpu.atmosphere import Atmosphere
from gpu.surface import SmoothWaterSurface
from cpu.cloudiness import Cloudiness3D
import gpu.satellite as satellite
# from cpu.utils import map2d
from cpu.atmosphere import Atmosphere as cpuAtm
from cpu.atmosphere import avg
from cpu.weight_funcs import krho
from cpu.core.static.weight_funcs import kw
import gpu.core.math as math


if __name__ == '__main__':

    THETA = 0.

    #########################################################################
    # domain parameters
    H = 20.  # высота атмосферы
    d = 1000  # дискретизация по высоте
    X = 10
    res = 10  # горизонтальная дискретизация

    # observation parameters
    # angle = 0.
    angle = THETA * np.pi / 180.  # зенитный угол наблюдения, по умолчанию: 0
    incline = 'left'
    # incline = 'right'
    integration_method = 'boole'

    # atmosphere parameters
    T0 = 15.
    P0 = 1013
    rho0 = 7.5

    # surface parameters
    surface_temperature = 15.
    surface_salinity = 0.

    # radiation parameters
    polarization = None
    frequencies = [22.2, 27.2, 36, 89]
    frequency_pairs = [(frequencies[0], frequencies[n]) for n in range(1, len(frequencies))]

    # # create project folder
    # folder = 'pre_res100_theta{}'.format(str(int(np.round(angle / np.pi * 180., decimals=0))).zfill(2))
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    #########################################################################

    atmosphere = Atmosphere.Standard(H=H, dh=H / d, T0=T0, P0=P0, rho0=rho0)  # для облачной атмосферы по Планку
    solid = Atmosphere.Standard(H=H, dh=H / d, T0=T0, P0=P0, rho0=rho0)  # для атмосферы со сплошной облачностью

    atmosphere.integration_method = integration_method  # метод интегрирования
    solid.integration_method = atmosphere.integration_method

    atmosphere.angle = angle
    atmosphere.horizontal_extent = X
    atmosphere.incline = incline  # в какую сторону по Ox наклонена траектория наблюдения
    solid.angle = atmosphere.angle
    solid.horizontal_extent = atmosphere.horizontal_extent
    solid.incline = atmosphere.incline

    surface = SmoothWaterSurface(temperature=surface_temperature,
                                 salinity=surface_salinity,
                                 polarization=polarization)  # модель гладкой водной поверхности
    surface.angle = atmosphere.angle

    #########################################################################
    # Параметры для моделирования высотного распределения водности

    const_w = False
    mu0 = 3.27
    psi0 = 0.67

    _c0 = 0.132574
    _c1 = 2.30215

    cl_bottom = 1.

    #########################################################################
    # precomputes
    sa = cpuAtm.Standard(T0, P0, rho0)
    T_avg_down, T_avg_up, Tau_o, A, B, M = {}, {}, {}, {}, {}, {}
    T_cosmic = 2.72548
    for i, freq_pair in enumerate(frequency_pairs):
        k_rho = [krho(sa, f) for f in freq_pair]
        k_w = math.as_tensor([kw(f, t=0.) for f in freq_pair])
        m = math.as_tensor([k_rho, k_w])
        M[i] = math.transpose(m)

        Tau_o[i] = math.as_tensor([sa.opacity.oxygen(nu) for nu in freq_pair])

        T_avg_down[i] = math.as_tensor([avg.downward.T(sa, nu, angle) for nu in freq_pair])
        T_avg_up[i] = math.as_tensor([avg.upward.T(sa, nu, angle) for nu in freq_pair])

        R = math.as_tensor([surface.reflectivity(nu) for nu in freq_pair])
        kappa = 1 - R
        A[i] = (T_avg_down[i] - T_cosmic) * R
        B[i] = T_avg_up[i] - T_avg_down[i] * R - math.as_tensor(surface_temperature + 273.15) * kappa

    #########################################################################

    data = [('power', 'procentage', 'Q_mean', 'W_mean',
             'freq_pair_no', 'nu1', 'nu2',
             'tb_mean_nu1', 'tb_mean_nu2',
             'tau_mean_nu1', 'tau_mean_nu2',
             'Qr_mean', 'Wr_mean', 'Qrs', 'Wrs')]

    # По порядку:
    # 1) мощность (высота) облака в ячейке
    # 2) процент заполнения = кол-во облачных ячеек / общее кол-во ячеек (100) * 100.
    # 3) истинное среднее значение водяного пара по всей расчетной области, интеграл от профиля абс. влажности по высоте
    # 4) истинное среднее по всей расчетной области значение водозапаса облаков
    # 5)порядковый номер комбинации частотных каналов
    # 6) частота №1
    # 7) частота №2
    # 8) среднее по всей расчетной области значение яркостной температуры для частоты 1
    # 9) среднее по всей расчетной области значение яркостной температуры для частоты 2
    # 10) среднее по всей расчетной области значение коэффициента полного поглощения для частоты 1
    # 11) среднее по всей расчетной области значение коэффициента полного поглощения для частоты 2
    # 12-13) Решим обратную задачу двухчастотным методом согласно, получим оценки на восстановленные значения
    # $Q$ и $W$ по каждому узлу расчетной сети, причем в качестве первой из частот будем всегда выбирать канал 22.2 ГГц
    # (близкий к резонансной линии водяного пара 22.235 ГГц), а в качестве второй - 27.2, 36 либо 89 ГГц.
    # В результате расчета получим три пары 2D-карт восстановленных значений TWV и LWC. Для каждой из таких карт
    # найдем соответствующее среднее значение. Таким образом, сначала решается обратная задача, затем полученные
    # карты TWV и LWC усредняются.
    # 14-15) В каждом канале 22.2, 27.2, 36 и 89 ГГц найдем среднее значение Тя по всей расчетной области.
    # Используя эти четыре значения снова решим обратную задачу и восстановим TWV и LWC, выбирая те же комбинации
    # частот, что и выше. Заметим, что операция усреднения здесь применяется напрямую к картам яркостных температур,
    # то есть ДО решения обратной задачи.

    #####################
    power_range = np.linspace(0, 5., 40)[::-1]
    #####################

    for pl, power in enumerate(power_range):

        hmap = np.zeros((res, res))

        for i in range(0, res, 2):
            for j in range(0, res, 2):

                hmap[i, j] = power
                hmap[i, j+1] = power
                hmap[i+1, j] = power
                hmap[i+1, j+1] = power

                procentage = np.count_nonzero(hmap) / (res * res) * 100

                print('\rpower = {:.1f}\tfill = {:.1f}%\t'.format(
                      power, procentage),
                      end='     ', flush=True)

                c = Cloudiness3D(kilometers=(X, X, H), nodes=(res, res, d), clouds_bottom=cl_bottom)

                atmosphere.liquid_water = c.liquid_water(height_map2d=hmap,
                                                         const_w=const_w, mu0=mu0, psi0=psi0,
                                                         _w=lambda _H: _c0 * np.power(_H, _c1))

                brts, brts_mean = {}, {}
                taus, taus_mean = {}, {}
                for nu in frequencies:
                    brt = satellite.brightness_temperature(nu, atmosphere, surface, cosmic=True)
                    brt = np.asarray(brt, dtype=float)
                    brts[nu] = brt
                    brts_mean[nu] = np.mean(brt)

                    tau = atmosphere.opacity.summary(nu)
                    tau = np.asarray(tau, dtype=float)
                    taus[nu] = tau
                    taus_mean[nu] = np.mean(tau)

                Q = atmosphere.Q
                Q = np.asarray([[Q] * res] * res)

                nx, ny = brts[frequencies[0]].shape
                if incline == 'left':
                    W = atmosphere.W[res - nx:, :]
                    Q = Q[res - nx:, :]
                else:
                    W = atmosphere.W[:nx, :]
                    Q = Q[:nx, :]

                W_mean = np.mean(W)
                Q_mean = np.mean(Q)

                for k, (nu1, nu2) in enumerate(frequency_pairs):
                    a, b = A[k], B[k]
                    t_avg_up = T_avg_up[k]
                    mat = M[k]
                    tau_o = Tau_o[k]

                    brt = math.as_tensor([brts[nu1].flatten(), brts[nu2].flatten()])
                    brt = math.move_axis(brt, 0, -1)

                    D = b * b - 4 * a * (brt - t_avg_up)
                    tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)

                    right = math.move_axis(tau_e - tau_o, 0, -1)

                    sol = math.linalg_solve(mat, right)
                    Wr = np.asarray(sol[1, :], dtype=float)
                    Qr = np.asarray(sol[0, :], dtype=float)

                    Wr_mean = np.mean(Wr)
                    Qr_mean = np.mean(Qr)

                    #####################

                    brt = math.as_tensor([brts_mean[nu1], brts_mean[nu2]])
                    D = b * b - 4 * a * (brt - t_avg_up)
                    tau_e = np.asarray(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)
                    sol = np.linalg.solve(mat, tau_e - tau_o)
                    Wrs = sol[1]
                    Qrs = sol[0]

                    #####################

                    data.append([power, procentage, Q_mean, W_mean,
                                 k, nu1, nu2,
                                 brts_mean[nu1], brts_mean[nu2],
                                 taus_mean[nu1], taus_mean[nu2],
                                 Qr_mean, Wr_mean, Qrs, Wrs])

        with open('pre_data.bin', 'wb') as dump:
            dill.dump(np.array(data, dtype=object), dump, recurse=True)

    data = np.array(data, dtype=object)
    with open('pre_data.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)
