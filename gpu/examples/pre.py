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


if __name__ == '__main__':

    THETA = 0.

    #########################################################################
    # domain parameters
    H = 20.  # высота атмосферы
    d = 100  # дискретизация по высоте
    X = 10
    res = 10  # горизонтальная дискретизация

    # observation parameters
    # angle = 0.
    angle = THETA * np.pi / 180.  # зенитный угол наблюдения, по умолчанию: 0
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

    # create project folder
    folder = 'pre_res100_theta{}'.format(str(int(np.round(angle / np.pi * 180., decimals=0))).zfill(2))
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

    const_w = False
    mu0 = 3.27
    psi0 = 0.67

    _c0 = 0.132574
    _c1 = 2.30215

    cl_bottom = 1.5

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

    data = [('power', 'procentage', 'Q_mean', 'W_mean',
             'freq_pair_no', 'nu1', 'nu2', 'Qr_mean', 'Wr_mean', 'Qrs', 'Wrs')]

    #####################
    power_range = np.linspace(0, 5., 20)[::-1]
    #####################

    for pl, power in enumerate(power_range):

        hmap = np.zeros((res, res))

        for i in range(res):
            for j in range(res):

                hmap[i, j] = power
                procentage = np.count_nonzero(hmap) / (res * res) * 100

                print('\r{:.2f}%\t\t\tpower = {:.1f}\tfill = {:.1f}%\t'.format(
                      (i * res + j + 1) * (pl + 1) / (res * res * len(power_range)) * 100,
                      power, procentage),
                      end='     ', flush=True)

                c = Cloudiness3D(kilometers=(X, X, H), nodes=(res, res, d), clouds_bottom=cl_bottom)

                atmosphere.liquid_water = c.liquid_water(height_map2d=hmap,
                                                         const_w=const_w, mu0=mu0, psi0=psi0,
                                                         _w=lambda _H: _c0 * np.power(_H, _c1))

                brts, brts_mean = {}, {}
                # taus, taus_mean = {}, {}
                for nu in frequencies:
                    brt = satellite.brightness_temperature(nu, atmosphere, surface, cosmic=True)
                    brt = np.asarray(brt, dtype=float)
                    brts[nu] = brt
                    brts_mean[nu] = np.mean(brt)

                    # tau = atmosphere.opacity.summary(nu)
                    # tau = np.asarray(tau, dtype=float)
                    # taus[nu] = tau
                    # taus_mean[nu] = np.mean(tau)

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
                                 k, nu1, nu2, Qr_mean, Wr_mean, Qrs, Wrs])

    data = np.array(data, dtype=object)
    with open('pre_data.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)
