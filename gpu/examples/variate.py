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
import gpu.core.math as math

from matplotlib import pyplot as plt


def new_stats():
    return {
        'mean': defaultdict(list), 'min': defaultdict(list), 'max': defaultdict(list),
        'var': defaultdict(list), 'std': defaultdict(list), 'range': defaultdict(list),
    }


def append_stats(dest, arr, key):
    dest['mean'][key].append(np.mean(arr))
    dest['min'][key].append(np.min(arr))
    dest['max'][key].append(np.max(arr))
    dest['var'][key].append(np.var(arr))
    dest['std'][key].append(np.std(arr))
    dest['range'][key].append(np.max(arr) - np.min(arr))


if __name__ == '__main__':

    #########################################################################
    # domain parameters
    H = 20.  # высота атмосферы
    d = 100  # дискретизация по высоте
    X = 50  # (км) горизонтальная протяженность моделируемой атмосферной ячейки
    res = 300  # горизонтальная дискретизация

    # observation parameters
    # angle = 0.
    angle = 30. * np.pi / 180.             # зенитный угол наблюдения, по умолчанию: 0
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
    # frequencies = [22.2, 27.2]  # DEBUG

    kernels = [int(a) for a in np.arange(6, 294+1, 6)]
    #########################################################################

    # create project folder
    folder = 'ieee_{}'.format(str(np.round(angle / np.pi * 180., decimals=0)).zfill(3))
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
    for i, freq_pair in enumerate([(frequencies[0], frequencies[n]) for n in range(1, len(frequencies))]):
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
    percentage = np.linspace(0.2, 0.7, 20, endpoint=True)[::-1]
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
            brts = []
            for nu in frequencies:

                brt = satellite.brightness_temperature(nu, atmosphere, surface, cosmic=True)

                brt = np.asarray(brt, dtype=float)
                brts.append(brt)

            # W = atmosphere.W
            nx, ny = brts[0].shape
            if incline == 'left':
                W = atmosphere.W[res-nx:, :]
            else:
                W = atmosphere.W[:nx, :]

            # # =========== TEST ===========
            # print('\n\n=========== TEST ===========\n')
            # plt.figure('nu = {:.1f}'.format(frequencies[0]))
            # plt.imshow(brts[0])
            # plt.colorbar()
            # print('BRT SHAPE : ', brts[0].shape)
            # plt.figure('W INITIAL')
            # plt.imshow(W)
            # plt.colorbar()
            # print('W SHAPE : ', W.shape)
            #
            # i, nu1, nu2 = 0, frequencies[0], frequencies[1]
            # a, b = A[i], B[i]
            # t_avg_up = T_avg_up[i]
            # mat = M[i]
            # tau_o = Tau_o[i]
            # _brt = math.as_tensor([brts[0].flatten(), brts[1].flatten()])
            # _brt = math.move_axis(_brt, 0, -1)
            # D = b * b - 4 * a * (_brt - t_avg_up)
            # tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)
            #
            # right = math.move_axis(tau_e - tau_o, 0, -1)
            #
            # sol = math.linalg_solve(mat, right)
            # wbrt = np.asarray(sol[1, :], dtype=float)
            # print('wbrt RETR SHAPE : ', wbrt.shape)
            # plt.figure('W RETR')
            # plt.imshow(np.reshape(wbrt, (nx, ny)))
            # plt.colorbar()
            # plt.show()
            # exit(0)
            # # =========== END TEST ===========

            WINI = {'mean': [], 'min': [], 'max': [], 'var': [], 'std': [], 'range': []}

            # template = {
            #     'mean': defaultdict(list), 'min': defaultdict(list), 'max': defaultdict(list),
            #     'var': defaultdict(list), 'std': defaultdict(list), 'range': defaultdict(list),
            # }

            BRTC = new_stats()
            SOLD = new_stats()
            DTSB = new_stats()

            WBRT = new_stats()
            WSOL = new_stats()
            DWSB = new_stats()
            DWBI = new_stats()
            DWSI = new_stats()
            DWSBI = new_stats()
            DWBII = new_stats()
            DWSII = new_stats()

            print('Making convolution...')
            for kernel in kernels:
                elapsed = datetime.datetime.now() - start_time
                days = elapsed.days
                hours = elapsed.seconds // 3600
                minutes = (elapsed.seconds - hours * 3600) // 60
                seconds = elapsed.seconds - hours * 3600 - minutes * 60

                print('\rkernel: ({}, {})\t\t\t{} d\t{} h\t{} m\t{} s'.format(kernel, kernel,
                                                                            days, hours, minutes, seconds),
                      end='   ', flush=True)

                # свертка карты водозапаса с элементом разрешения выбранного размера
                conv_w = map2d.conv_averaging(W, kernel=kernel)
                WINI['mean'].append(np.mean(conv_w))
                WINI['min'].append(np.min(conv_w))
                WINI['max'].append(np.max(conv_w))
                WINI['var'].append(np.var(conv_w))
                WINI['std'].append(np.std(conv_w))
                WINI['range'].append(np.max(conv_w) - np.min(conv_w))

                # обратный переход от водозапаса к высотам с учетом сделанной ранее коррекции
                conv_h = np.power(conv_w / _c0, 1. / _c1)

                solid.liquid_water = Cloudiness3D(kilometers=(res // X, res // X * len(conv_h), H),
                                                  nodes=(1, len(conv_h), d), clouds_bottom=cl_bottom).liquid_water(
                    np.asarray([conv_h]), const_w=False, _w=lambda _H: _c0 * np.power(_H, _c1)
                )

                conv_brts = {}
                solid_brts = {}
                for j, nu in enumerate(frequencies):
                    conv_brt = map2d.conv_averaging(brts[j], kernel=kernel)
                    append_stats(BRTC, conv_brt, nu)
                    conv_brts[nu] = conv_brt

                    solid_brt = satellite.brightness_temperature(nu, solid, surface, cosmic=True, __theta=angle)[0]
                    solid_brt = np.asarray(solid_brt, dtype=float)
                    append_stats(SOLD, solid_brt, nu)
                    solid_brts[nu] = solid_brt

                    delta = solid_brt - conv_brt
                    append_stats(DTSB, delta, nu)

                append_stats(WBRT, [sys.maxsize], frequencies[0])
                append_stats(WSOL, [sys.maxsize], frequencies[0])
                append_stats(DWSB, [sys.maxsize], frequencies[0])
                append_stats(DWBI, [sys.maxsize], frequencies[0])
                append_stats(DWSI, [sys.maxsize], frequencies[0])
                append_stats(DWSBI, [sys.maxsize], frequencies[0])
                append_stats(DWBII, [sys.maxsize], frequencies[0])
                append_stats(DWSII, [sys.maxsize], frequencies[0])
                for i, (nu1, nu2) in enumerate([(frequencies[0], frequencies[j]) for j in range(1, len(frequencies))]):
                    a, b = A[i], B[i]
                    t_avg_up = T_avg_up[i]
                    mat = M[i]
                    tau_o = Tau_o[i]
                    conv_brt = math.as_tensor([conv_brts[nu1], conv_brts[nu2]])
                    conv_brt = math.move_axis(conv_brt, 0, -1)

                    D = b * b - 4 * a * (conv_brt - t_avg_up)
                    tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)

                    right = math.move_axis(tau_e - tau_o, 0, -1)

                    sol = math.linalg_solve(mat, right)
                    wbrt = np.asarray(sol[1, :], dtype=float)
                    append_stats(WBRT, wbrt, nu2)

                    ##################################################################################################
                    solid_brt = np.asarray([solid_brts[nu1], solid_brts[nu2]])
                    solid_brt = np.moveaxis(solid_brt, 0, -1)

                    D = b * b - 4 * a * (solid_brt - t_avg_up)
                    tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)

                    right = math.move_axis(tau_e - tau_o, 0, -1)

                    sol = math.linalg_solve(mat, right)
                    wsolid = np.asarray(sol[1, :], dtype=float)
                    append_stats(WSOL, wsolid, nu2)

                    append_stats(DWSB, wsolid - wbrt, nu2)
                    append_stats(DWBI, wbrt - conv_w, nu2)
                    append_stats(DWSI, wsolid - conv_w, nu2)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        delta = np.where(np.isclose(conv_w, 0.), sys.maxsize, (wsolid - wbrt) / conv_w)
                        append_stats(DWSBI, delta, nu2)

                        delta = np.where(np.isclose(conv_w, 0.), sys.maxsize, (wbrt - conv_w) / conv_w)
                        append_stats(DWBII, delta, nu2)

                        delta = np.where(np.isclose(conv_w, 0.), sys.maxsize, (wsolid - conv_w) / conv_w)
                        append_stats(DWSII, delta, nu2)

            data = {
                'name': distr['name'],
                'part': ID,
                'filename': '{}_{}.part'.format(distr['name'], str(ID).zfill(4)),

                'H': H,
                'd': d,
                'X': X,
                'res': res,

                'angle': angle,
                'incline': incline,
                'integration_method': integration_method,
                'T0': T0,
                'P0': P0,
                'rho0': rho0,
                'surface_temperature': surface_temperature,
                'surface_salinity': surface_salinity,

                'required_percentage': np.round(required_percentage * 100., decimals=3),
                'cover_percentage': np.round(cover_percentage * 100., decimals=3),
                'cover_percentage_d': np.round(cover_percentage_d * 100., decimals=3),
                'sky_cover': sky_cover,
                'sky_cover_d': sky_cover_d,

                'seed': seed,

                'K': np.round(K, decimals=3),
                'alpha': alpha,
                'Dm': Dm,
                'd_min': dm,
                'xi': xi,

                'n_analytical': N_analytical,
                'n_fact': N_fact,

                'eta': eta,
                'beta': beta,
                'cl_bottom': cl_bottom,

                # 'clouds': clouds,
                # 'heights': {
                #     'map': hmap,
                # },

                'mu0': mu0,
                'psi0': psi0,
                'c0': _c0,
                'c1': _c1,

                # 'liquid_water': atmosphere.liquid_water,

                'kernels': kernels,

                'W': {
                    # 'map': W,
                    'total_max': np.max(W),
                    'total_mean': np.mean(W),
                    'total_min': np.min(W),
                    'total_var': np.var(W),
                    'total_std': np.std(W),
                    'total_range': np.max(W) - np.min(W),

                    'WINI': WINI,

                    'WBRT': WBRT,
                    'WSOL': WSOL,
                    'DWSB': DWSB,
                    'DWBI': DWBI,
                    'DWSI': DWSI,

                    'DWSBI': DWSBI,
                    'DWBII': DWBII,
                    'DWSII': DWSII,
                },

                'frequencies': frequencies,

                'brightness_temperature': {
                    # 'maps': brts,
                    'BRTC': BRTC,
                    'SOLD': SOLD,
                    'DTSB': DTSB,
                }
            }

            with open(os.path.join(folder, data['filename']), 'wb') as dump:
                dill.dump(data, dump)
