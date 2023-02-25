# -*- coding: utf-8 -*-
import os
# import sys
# import warnings
import dill
import datetime
import numpy as np
# from collections import defaultdict

from gpu.atmosphere import Atmosphere
from gpu.surface import SmoothWaterSurface
from cpu.cloudiness import Plank3D, Cloudiness3D
import gpu.satellite as satellite
from cpu.utils import map2d
from cpu.atmosphere import Atmosphere as cpuAtm
from cpu.atmosphere import avg
from cpu.weight_funcs import krho
from cpu.core.static.weight_funcs import kw
import cpu.core.math as math


class Stat:
    def __init__(self, variable):
        self.mean = np.mean(variable)
        self.min = np.min(variable)
        self.max = np.max(variable)
        self.var = np.var(variable)
        self.std = np.std(variable)


if __name__ == '__main__':

    # THETAS = [0., 10., 20., 30., 40., 51.]
    THETAS = [0.]

    data = [(
        'angle',    # угол наблюдения
        'distr_no',     # порядковый номер планковского распределения
        'distr_name',    # название распределения
        'alpha', 'Dm', 'dm', 'eta', 'beta', 'cl_bottom',    # параметры распределения
        'xi', 'K',    # ещё параметры (K зависит от %)
        'required_percentage',    # % покрытия облаками (cloud amount)

        'Q_TRUE', 'W_TRUE',    # средние по всей расчетной сетке

        'kernel',    # ядро усреднения / размер эл-та разрешения радиометра в узлах

        'Q_true', 'W_true',    # статистика на истинные средние значения TWV и LWC
        'H_true'    # статистика на истинную вертикальную протяженность (высота верхней кромки -1)
        'efl_H',    # статистика на высоту эквивалентного по водозапасу сплошного слоя облачности

        'freq_pair_no', 'nu1', 'nu2',    # порядковый номер пары частот, сами частоты

        # статистика в элементе разрешения
        'tb_nu1', 'tb_nu2',    # яркостные температуры
        'tau_nu1', 'tau_nu2',    # коэффициент полного поглощения
        'efl_tb_nu1', 'efl_tb_nu2',    # яркостная температура для эквивалентного по водозапасу сплошного слоя
        'efl_tau_nu1', 'efl_tau_nu2',    # полное поглощения для эквивалентного по водозапасу сплошного слоя

        # статистика в элементе разрешения - восстановление TWV и LWC
        'Qr', 'Wr',    # сначала решение обратной задачи, затем усреднение полученных 2D-карт TWV и LWC
        'Qrs', 'Wrs',    # сначала усреднение 2D-карт яркостной температуры, затем решение обратной задачи
        'Qrss', 'Wrss',    # TWV и LWC, восстановленные по яркостной температуре эквивалентного сплошного слоя

        # Дополнительно
        'Delta_Qr', 'Delta_Wr',
        'Delta_Qrs', 'Delta_Wrs',
        'Delta_Qrss', 'Delta_Wrss',

        'relerr_Qr', 'relerr_Wr',
        'relerr_Qrs', 'relerr_Wrs',
        'relerr_Qrss', 'relerr_Wrss',

        'Delta_Qrs_Qr', 'Delta_Wrs_Wr',
        'Delta_Qrs_Qrss', 'Delta_Wrs_Wrss',
        'Delta_Qr_Qrss', 'Delta_Wr_Wrss',
    )]

    for THETA in THETAS:

        #########################################################################
        # domain parameters
        H = 20.  # высота атмосферы
        d = 100  # дискретизация по высоте
        X = 50  # (км) горизонтальная протяженность моделируемой атмосферной ячейки
        res = 300  # горизонтальная дискретизация

        # observation parameters
        # angle = 0.
        angle = THETA * np.pi / 180.             # зенитный угол наблюдения, по умолчанию: 0
        incline = 'left'
        # incline = 'right'
        integration_method = 'trapz'

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

        # kernels = [int(a) for a in np.arange(6, res+1, 6)]
        kernels = [60]
        # kernels = list(range(6, 96, 6)) + [120, 150, 240, 288]
        #########################################################################

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
        # base distribution parameters
        base_distributions = [

            {'name': 'L2', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 0.93, 'beta': 0.3, 'cl_bottom': 1.2192},

            {'name': 'L3', 'alpha': 1.485, 'Dm': 4.020, 'dm': 0.03048, 'eta': 0.76, 'beta': -0.3, 'cl_bottom': 1.3716},
            {'name': 'T7', 'alpha': 1.35, 'Dm': 3.733, 'dm': 0.04572, 'eta': 1.2, 'beta': 0.0, 'cl_bottom': 1.24968},
            {'name': 'T6', 'alpha': 1.398, 'Dm': 3.376, 'dm': 0.03048, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 1.0668},
            {'name': 'T8', 'alpha': 1.485, 'Dm': 4.02, 'dm': 0.06096, 'eta': 1.2, 'beta': 0.4, 'cl_bottom': 1.3716},
            {'name': 'T9', 'alpha': 2.485, 'Dm': 2.656, 'dm': 0.04572, 'eta': 1.3, 'beta': 0.3, 'cl_bottom': 1.40208},
            #
            {'name': 'L1', 'alpha': 3.853, 'Dm': 1.448, 'dm': 0.01524, 'eta': 0.98, 'beta': 0.0, 'cl_bottom': 0.54864},
            {'name': 'T5', 'alpha': 2.051, 'Dm': 2.574, 'dm': 0.02286, 'eta': 0.85, 'beta': -0.13, 'cl_bottom': 1.11252},
            {'name': 'T3', 'alpha': 2.361, 'Dm': 2.092, 'dm': 0.01524, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 0.82296},
            {'name': 'T4', 'alpha': 2.703, 'Dm': 2.094, 'dm': 0.02286, 'eta': 0.8, 'beta': 0.0, 'cl_bottom': 0.9144},
            {'name': 'T2', 'alpha': 4.412, 'Dm': 1.126, 'dm': 0.01524, 'eta': 0.97, 'beta': 0.0, 'cl_bottom': 0.70104},
            {'name': 'T1', 'alpha': 9.07, 'Dm': 0.80485, 'dm': 0.01524, 'eta': 0.89, 'beta': 0.0, 'cl_bottom': 0.67056},
        ]
        seed = 42

        const_w = False
        mu0 = 3.27
        psi0 = 0.67

        _c0 = 0.132574
        _c1 = 2.30215

        #########################################################################
        # precomputes
        sa = cpuAtm.Standard(T0, P0, rho0)
        T_avg_down, T_avg_up, Tau_o, A, B, M = {}, {}, {}, {}, {}, {}
        T_cosmic = 2.72548
        for i, freq_pair in enumerate(frequency_pairs):
            k_rho = [krho(sa, f) for f in freq_pair]
            k_w = [kw(f, t=-5) for f in freq_pair]
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
        percentage = np.linspace(0.2, 0.7, 20, endpoint=True)[::-1]
        #########################################################################

        distributions = []
        for base_distr in base_distributions:
            distributions.append(base_distr)
            for i, beta in enumerate(np.arange(-0.9, 0.9, 0.1)):
                new_distr = base_distr.copy()
                new_distr['name'] += 'B' + str(i).zfill(2)
                new_distr['beta'] = np.round(beta, decimals=1)
                distributions.append(new_distr)
            for j, eta in enumerate(np.arange(0.53, 1.83, 0.1)):
                new_distr = base_distr.copy()
                new_distr['name'] += 'E' + str(j).zfill(2)
                new_distr['eta'] = np.round(eta, decimals=2)
                distributions.append(new_distr)

        start_time = datetime.datetime.now()

        for distr_no, distr in enumerate(distributions):
            print('\n\nProcessing {} ...\n'.format(distr['name']))

            alpha, Dm, dm, eta, beta, cl_bottom = \
                distr['alpha'], distr['Dm'], distr['dm'], distr['eta'], distr['beta'], distr['cl_bottom']

            xi = -np.exp(-alpha * Dm) * (((alpha * Dm) ** 2) / 2 + alpha * Dm + 1) + \
                np.exp(-alpha * dm) * (((alpha * dm) ** 2) / 2 + alpha * dm + 1)
            print('xi\t', xi)

            for ID, required_percentage in enumerate(percentage):
                # if os.path.exists(os.path.join(folder, '{}_{}.part'.format(distr['name'], str(ID).zfill(4)))):
                #     continue
                print('\n\nRequired %: {:.2f}'.format(required_percentage * 100.))
                K = 2 * np.power(alpha, 3) * (X * X * required_percentage) / (np.pi * xi)
                print('K\t', K)

                # print(os.getcwd())
                try:
                    with open(os.path.join('HMAP',
                                           distr['name'] + '_P' + str(
                                               int(np.round(required_percentage * 100., decimals=0))) + '.map'),
                              'rb') as file:
                        hmap = dill.load(file)
                except FileNotFoundError:
                    continue

                p = Plank3D(kilometers=(X, X, H), nodes=(res, res, d), clouds_bottom=cl_bottom)

                cover_percentage_d = np.count_nonzero(hmap) / (res * res)
                sky_cover_d = cover_percentage_d * (X * X)

                print('Simulating liquid water distribution 3D...')
                atmosphere.liquid_water = p.liquid_water_(hmap2d=hmap,
                                                          const_w=const_w, mu0=mu0, psi0=psi0,
                                                          _w=lambda _H: _c0 * np.power(_H, _c1))
                # print('LW3D shape: {}'.format(atmosphere.liquid_water.shape))

                print('Calculating brightness temperatures...')
                brts = {}
                taus = {}
                for nu in frequencies:

                    brt = satellite.brightness_temperature(nu, atmosphere, surface, cosmic=True)

                    brt = np.asarray(brt, dtype=float)
                    brts[nu] = brt

                    tau = atmosphere.opacity.summary(nu)
                    tau = np.asarray(tau, dtype=float)
                    taus[nu] = tau

                # W = atmosphere.W
                Q = atmosphere.Q
                Q = np.asarray([[Q] * res] * res)

                nx, ny = brts[frequencies[0]].shape
                if incline == 'left':
                    W = atmosphere.W[res-nx:, :]
                    Q = Q[res-nx:, :]
                else:
                    W = atmosphere.W[:nx, :]
                    Q = Q[:nx, :]

                print('Making convolution...')
                for kernel in kernels:

                    if kernel > nx:
                        continue

                    elapsed = datetime.datetime.now() - start_time
                    days = elapsed.days
                    hours = elapsed.seconds // 3600
                    minutes = (elapsed.seconds - hours * 3600) // 60
                    seconds = elapsed.seconds - hours * 3600 - minutes * 60

                    print('\rkernel: ({}, {})\t\t\t{} d\t{} h\t{} m\t{} s'.format(kernel, kernel,
                                                                                  days, hours, minutes, seconds),
                          end='   ', flush=True)

                    conv_H_mean = map2d.conv_averaging(hmap, kernel=kernel)

                    # свертка карты водозапаса с элементом разрешения выбранного размера
                    conv_W_mean = map2d.conv_averaging(W, kernel=kernel)
                    conv_Q_mean = map2d.conv_averaging(Q, kernel=kernel)

                    # обратный переход от водозапаса к высотам с учетом сделанной ранее коррекции
                    conv_Hs = np.power(conv_W_mean / _c0, 1. / _c1)

                    solid.liquid_water = Cloudiness3D(kilometers=(res // X, res // X * len(conv_Hs), H),
                                                      nodes=(1, len(conv_Hs), d), clouds_bottom=cl_bottom).liquid_water(
                        np.asarray([conv_Hs]), const_w=False, _w=lambda _H: _c0 * np.power(_H, _c1)
                    )

                    conv_brts_mean = {}
                    conv_taus_mean = {}
                    solid_brts = {}
                    solid_taus = {}
                    for nu in frequencies:
                        conv_brt_mean = map2d.conv_averaging(brts[nu], kernel=kernel)
                        conv_brts_mean[nu] = conv_brt_mean

                        conv_tau_mean = map2d.conv_averaging(taus[nu], kernel=kernel)
                        conv_taus_mean[nu] = conv_tau_mean

                        solid_brt = satellite.brightness_temperature(nu, solid, surface, cosmic=True, __theta=angle)[0]
                        solid_brt = np.asarray(solid_brt, dtype=float)
                        solid_brts[nu] = solid_brt

                        solid_tau = solid.opacity.summary(nu, angle)[0]
                        solid_tau = np.asarray(solid_tau, dtype=float)
                        solid_taus[nu] = solid_tau

                    for i, (nu1, nu2) in enumerate(frequency_pairs):
                        a, b = A[i], B[i]
                        t_avg_up = T_avg_up[i]
                        mat = M[i]
                        tau_o = Tau_o[i]

                        brt = math.as_tensor([brts[nu1], brts[nu2]])
                        brt = math.move_axis(brt, 0, -1)

                        D = b * b - 4 * a * (brt - t_avg_up)
                        tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)

                        right = math.move_axis(tau_e - tau_o, 0, -1)

                        sol = math.linalg_solve(mat, right)
                        sol = math.move_axis(sol, 0, -1)
                        Wr = np.asarray(sol[1, :, :], dtype=float)
                        Qr = np.asarray(sol[0, :, :], dtype=float)

                        conv_Wr_mean = map2d.conv_averaging(Wr, kernel=kernel)
                        conv_Qr_mean = map2d.conv_averaging(Qr, kernel=kernel)

                        ###############################################################################################

                        conv_brt = math.as_tensor([conv_brts_mean[nu1], conv_brts_mean[nu2]])
                        conv_brt = math.move_axis(conv_brt, 0, -1)

                        D = b * b - 4 * a * (conv_brt - t_avg_up)
                        tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)

                        right = math.move_axis(tau_e - tau_o, 0, -1)

                        sol = math.linalg_solve(mat, right)
                        conv_Wrs = np.asarray(sol[1, :], dtype=float)
                        conv_Qrs = np.asarray(sol[0, :], dtype=float)

                        ###############################################################################################
                        solid_brt = np.asarray([solid_brts[nu1], solid_brts[nu2]])
                        solid_brt = np.moveaxis(solid_brt, 0, -1)

                        D = b * b - 4 * a * (solid_brt - t_avg_up)
                        tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)

                        right = math.move_axis(tau_e - tau_o, 0, -1)

                        sol = math.linalg_solve(mat, right)
                        conv_Wrss = np.asarray(sol[1, :], dtype=float)
                        conv_Qrss = np.asarray(sol[0, :], dtype=float)

                        ###############################################################################################

                        data.append(
                            [THETA,
                             distr_no,
                             distr['name'],
                             alpha, Dm, dm, eta, beta, cl_bottom,
                             xi, K,
                             required_percentage,

                             np.mean(Q), np.mean(W),

                             kernel,

                             Stat(conv_Q_mean), Stat(conv_W_mean),
                             Stat(conv_H_mean),
                             Stat(conv_Hs),

                             i, nu1, nu2,

                             Stat(conv_brts_mean[nu1]), Stat(conv_brts_mean[nu2]),
                             Stat(conv_taus_mean[nu1]), Stat(conv_taus_mean[nu2]),
                             Stat(solid_brts[nu1]), Stat(solid_brts[nu2]),
                             Stat(solid_taus[nu1]), Stat(solid_taus[nu2]),

                             Stat(conv_Qr_mean), Stat(conv_Wr_mean),
                             Stat(conv_Qrs), Stat(conv_Wrs),
                             Stat(conv_Qrss), Stat(conv_Wrss),

                             Stat(conv_Q_mean - conv_Qr_mean), Stat(conv_W_mean - conv_Wr_mean),
                             Stat(conv_Q_mean - conv_Qrs), Stat(conv_W_mean - conv_Wrs),
                             Stat(conv_Q_mean - conv_Qrss), Stat(conv_W_mean - conv_Wrss),

                             Stat((conv_Q_mean - conv_Qr_mean) / conv_Q_mean), Stat((conv_W_mean - conv_Wr_mean) / conv_W_mean),
                             Stat((conv_Q_mean - conv_Qrs) / conv_Q_mean), Stat((conv_W_mean - conv_Wrs) / conv_W_mean),
                             Stat((conv_Q_mean - conv_Qrss) / conv_Q_mean), Stat((conv_W_mean - conv_Wrss) / conv_W_mean),

                             Stat(conv_Qr_mean - conv_Qrs), Stat(conv_Wr_mean - conv_Wrs),
                             Stat(conv_Qrss - conv_Qrs), Stat(conv_Wrss - conv_Wrs),
                             Stat(conv_Qrss - conv_Qr_mean), Stat(conv_Wrss - conv_Wr_mean),
                             ]
                        )

            with open('post_data_theta0_kernel60_tcl-5_all.bin.part', 'wb') as dump:
                dill.dump(np.array(data, dtype=object), dump, recurse=True)

    data = np.array(data, dtype=object)
    with open('post_data_theta0_kernel60_tcl-5_all.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)

    os.remove('post_data_theta0_kernel60_tcl-5_all.bin.part')
