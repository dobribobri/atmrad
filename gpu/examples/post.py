# -*- coding: utf-8 -*-
import os
import sys
import warnings
import dill
import datetime
import numpy as np
from collections import defaultdict

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


if __name__ == '__main__':

    # THETAS = [0., 10., 20., 30., 40., 51.]
    THETAS = [0.]

    data = [
         'angle', 'distr', 'required_percentage', 'kernel',
            
         'Q_mean', 'W_mean',
         'efl_Hs_mean',
         'Q_max', 'W_max',
         'efl_Hs_max',
         'Q_min', 'W_min',
         'efl_Hs_min',
         'Q_var', 'W_var',
         'efl_Hs_var',
            
         'freq_pair_no', 'nu1', 'nu2',
            
         'tb_nu1_mean', 'tb_nu2_mean',
         'tau_nu1_mean', 'tau_nu2_mean',
         'efl_tb_nu1_mean', 'efl_tb_nu2_mean',
         'efl_tau_nu1_mean', 'efl_tau_nu2_mean',
         'Qr_mean', 'Wr_mean',
         'Qrs_mean', 'Wrs_mean',
         'efl_Qrss_mean', 'efl_Wrss_mean',

         'tb_nu1_max', 'tb_nu2_max',
         'tau_nu1_max', 'tau_nu2_max',
         'efl_tb_nu1_max', 'efl_tb_nu2_max',
         'efl_tau_nu1_max', 'efl_tau_nu2_max',
         'Qr_max', 'Wr_max',
         'Qrs_max', 'Wrs_max',
         'efl_Qrss_max', 'efl_Wrss_max'

         'tb_nu1_min', 'tb_nu2_min',
         'tau_nu1_min', 'tau_nu2_min',
         'efl_tb_nu1_min', 'efl_tb_nu2_min',
         'efl_tau_nu1_min', 'efl_tau_nu2_min',
         'Qr_min', 'Wr_min',
         'Qrs_min', 'Wrs_min',
         'efl_Qrss_min', 'efl_Wrss_min',

         'tb_nu1_var', 'tb_nu2_var',
         'tau_nu1_var', 'tau_nu2_var',
         'efl_tb_nu1_var', 'efl_tb_nu2_var',
         'efl_tau_nu1_var', 'efl_tau_nu2_var',
         'Qr_var', 'Wr_var',
         'Qrs_var', 'Wrs_var',
         'efl_Qrss_var', 'efl_Wrss_var',
    ]

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
        #########################################################################

        # create project folder
        folder = 'post_L2_kernel60_theta{}'.format(str(int(np.round(angle / np.pi * 180., decimals=0))).zfill(2))
        if not os.path.exists(folder):
            os.makedirs(folder)

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
        # distribution parameters
        distributions = [
            {'name': 'L2', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 0.93, 'beta': 0.3, 'cl_bottom': 1.2192},
            {'name': 'L2B', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 0.93, 'beta': -0.9, 'cl_bottom': 1.2192},
            {'name': 'L2E', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 1.5, 'beta': 0.3, 'cl_bottom': 1.2192},
            {'name': 'L2Z', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 1.5, 'beta': -0.9, 'cl_bottom': 1.2192},

            # {'name': 'L3', 'alpha': 1.485, 'Dm': 4.020, 'dm': 0.03048, 'eta': 0.76, 'beta': -0.3, 'cl_bottom': 1.3716},
            # {'name': 'T7', 'alpha': 1.35, 'Dm': 3.733, 'dm': 0.04572, 'eta': 1.2, 'beta': 0.0, 'cl_bottom': 1.24968},
            # {'name': 'T6', 'alpha': 1.398, 'Dm': 3.376, 'dm': 0.03048, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 1.0668},
            # {'name': 'T8', 'alpha': 1.485, 'Dm': 4.02, 'dm': 0.06096, 'eta': 1.2, 'beta': 0.4, 'cl_bottom': 1.3716},
            # {'name': 'T9', 'alpha': 2.485, 'Dm': 2.656, 'dm': 0.04572, 'eta': 1.3, 'beta': 0.3, 'cl_bottom': 1.40208},
            #
            # {'name': 'L1', 'alpha': 3.853, 'Dm': 1.448, 'dm': 0.01524, 'eta': 0.98, 'beta': 0.0, 'cl_bottom': 0.54864},
            # {'name': 'T5', 'alpha': 2.051, 'Dm': 2.574, 'dm': 0.02286, 'eta': 0.85, 'beta': -0.13, 'cl_bottom': 1.11252},
            # {'name': 'T3', 'alpha': 2.361, 'Dm': 2.092, 'dm': 0.01524, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 0.82296},
            # {'name': 'T4', 'alpha': 2.703, 'Dm': 2.094, 'dm': 0.02286, 'eta': 0.8, 'beta': 0.0, 'cl_bottom': 0.9144},
            # {'name': 'T2', 'alpha': 4.412, 'Dm': 1.126, 'dm': 0.01524, 'eta': 0.97, 'beta': 0.0, 'cl_bottom': 0.70104},
            # {'name': 'T1', 'alpha': 9.07, 'Dm': 0.80485, 'dm': 0.01524, 'eta': 0.89, 'beta': 0.0, 'cl_bottom': 0.67056},
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
            k_w = [kw(f, t=-2.) for f in freq_pair]
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
        percentage = np.linspace(0.2, 0.7, 10, endpoint=True)[::-1]
        #########################################################################

        start_time = datetime.datetime.now()

        for distr in distributions:
            print('\n\nProcessing {} ...\n'.format(distr['name']))

            alpha, Dm, dm, eta, beta, cl_bottom = \
                distr['alpha'], distr['Dm'], distr['dm'], distr['eta'], distr['beta'], distr['cl_bottom']

            xi = -np.exp(-alpha * Dm) * (((alpha * Dm) ** 2) / 2 + alpha * Dm + 1) + \
                np.exp(-alpha * dm) * (((alpha * dm) ** 2) / 2 + alpha * dm + 1)
            print('xi\t', xi)

            for ID, required_percentage in enumerate(percentage):
                if os.path.exists(os.path.join(folder, '{}_{}.part'.format(distr['name'], str(ID).zfill(4)))):
                    continue
                print('\n\nRequired %: {:.2f}'.format(required_percentage * 100.))
                K = 2 * np.power(alpha, 3) * (X * X * required_percentage) / (np.pi * xi)
                print('K\t', K)

                p = Plank3D(kilometers=(X, X, H), nodes=(res, res, d), clouds_bottom=cl_bottom)
                print('Generating cloud distribution...')
                try:
                    clouds = p.generate_clouds(
                        Dm=Dm, dm=dm, K=K, alpha=alpha, beta=beta, eta=eta, seed=seed, timeout=1., verbose=True
                    )
                except TimeoutError:
                    print('\n ...time is over')
                    continue

                N_analytical = K / alpha * (np.exp(-alpha * dm) - np.exp(-alpha * Dm))
                N_fact = len(clouds)
                print('N\tanalytical: {}\t\tactual: {}'.format(N_analytical, N_fact))

                hmap = p.height_map2d_(clouds)

                sky_cover = np.sum(np.pi * np.power(np.array([cloud.rx for cloud in clouds]), 2))
                cover_percentage = sky_cover / (X * X)
                cover_percentage_d = np.count_nonzero(hmap) / (res * res)
                sky_cover_d = cover_percentage_d * (X * X)
                # print('S\t before digitizing: {}\t\t after digitizing: {}'.format(sky_cover, sky_cover_d))
                print('%\tbefore digitizing: {}\t\tafter digitizing: {}'.format(
                    cover_percentage * 100., cover_percentage_d * 100.))

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
                            [THETA, distr, required_percentage, kernel,
                             np.mean(conv_Q_mean), np.mean(conv_W_mean),
                             np.mean(conv_Hs),
                             np.max(conv_Q_mean), np.max(conv_W_mean),
                             np.max(conv_Hs),
                             np.min(conv_Q_mean), np.min(conv_W_mean),
                             np.min(conv_Hs),
                             np.var(conv_Q_mean), np.var(conv_W_mean),
                             np.var(conv_Hs),
                             i, nu1, nu2,
                             np.mean(conv_brts_mean[nu1]), np.mean(conv_brts_mean[nu2]),
                             np.mean(conv_taus_mean[nu1]), np.mean(conv_taus_mean[nu2]), 
                             np.mean(solid_brts[nu1]), np.mean(solid_brts[nu2]),
                             np.mean(solid_taus[nu1]), np.mean(solid_taus[nu2]),
                             np.mean(conv_Qr_mean), np.mean(conv_Wr_mean),
                             np.mean(conv_Qrs), np.mean(conv_Wrs),
                             np.mean(conv_Qrss), np.mean(conv_Wrss),

                             np.max(conv_brts_mean[nu1]), np.max(conv_brts_mean[nu2]),
                             np.max(conv_taus_mean[nu1]), np.max(conv_taus_mean[nu2]),
                             np.max(solid_brts[nu1]), np.max(solid_brts[nu2]),
                             np.max(solid_taus[nu1]), np.max(solid_taus[nu2]),
                             np.max(conv_Qr_mean), np.max(conv_Wr_mean),
                             np.max(conv_Qrs), np.max(conv_Wrs),
                             np.max(conv_Qrss), np.max(conv_Wrss),

                             np.min(conv_brts_mean[nu1]), np.min(conv_brts_mean[nu2]),
                             np.min(conv_taus_mean[nu1]), np.min(conv_taus_mean[nu2]),
                             np.min(solid_brts[nu1]), np.min(solid_brts[nu2]),
                             np.min(solid_taus[nu1]), np.min(solid_taus[nu2]),
                             np.min(conv_Qr_mean), np.min(conv_Wr_mean),
                             np.min(conv_Qrs), np.min(conv_Wrs),
                             np.min(conv_Qrss), np.min(conv_Wrss),

                             np.var(conv_brts_mean[nu1]), np.var(conv_brts_mean[nu2]),
                             np.var(conv_taus_mean[nu1]), np.var(conv_taus_mean[nu2]),
                             np.var(solid_brts[nu1]), np.var(solid_brts[nu2]),
                             np.var(solid_taus[nu1]), np.var(solid_taus[nu2]),
                             np.var(conv_Qr_mean), np.var(conv_Wr_mean),
                             np.var(conv_Qrs), np.var(conv_Wrs),
                             np.var(conv_Qrss), np.var(conv_Wrss),
                             ]
                        )

    data = np.array(data, dtype=object)
    with open('post_data.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)
