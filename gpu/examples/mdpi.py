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

    data = [(
        'THETA',
        'h_true',
        'q_true', 'w_true', 'freq_pair_no', 'nu1', 'nu2',
        'Qr', 'Wr', 'Qrv', 'Wrv',
        'DQrv', 'DWrv', 'dQrv', 'dWrv',
    )]

    THETAS = [0., 10., 20., 30., 40., 51.]

    for THETA in THETAS:

        #########################################################################
        # domain parameters
        H = 20.  # высота атмосферы
        d = 500  # дискретизация по высоте
        X = 1  # (км) горизонтальная протяженность моделируемой атмосферной ячейки
        # res = 300  # горизонтальная дискретизация

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
        #########################################################################

        solid = Atmosphere.Standard(H=H, dh=H / d, T0=T0, P0=P0, rho0=rho0)  # для атмосферы со сплошной облачностью
        solid.integration_method = integration_method
        solid.angle = angle
        solid.horizontal_extent = X
        solid.incline = incline

        surface = SmoothWaterSurface(temperature=surface_temperature,
                                     salinity=surface_salinity,
                                     polarization=polarization)  # модель гладкой водной поверхности
        surface.angle = solid.angle

        #########################################################################

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
            k_w = [kw(f, t=0.) for f in freq_pair]
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

        H_true_range = np.arange(cl_bottom + 0.12, 5. + 0.12, 0.1)

        q_true = solid.Q

        for jh, h_true in enumerate(H_true_range):
            print('\rTHETA is {:.1f}, \t\t {:.2f}%'.format(
                THETA,
                (jh + 1) / len(H_true_range) * 100.),
                end='    ', flush=True)

            w_true = _c0 * np.power(h_true, _c1)

            height_range = np.insert(np.linspace(h_true * 0.9, h_true * 1.1, 20, endpoint=True), 0, h_true)

            solid.liquid_water = Cloudiness3D(kilometers=(1, 1 * len(height_range), H),
                                              nodes=(1, len(height_range), d), clouds_bottom=cl_bottom).liquid_water(
                np.asarray([height_range]), const_w=False, _w=lambda _H: _c0 * np.power(_H, _c1)
            )

            solid_brts = {}
            for nu in frequencies:
                solid_brt = satellite.brightness_temperature(nu, solid, surface, cosmic=True, __theta=angle)[0]
                solid_brt = np.asarray(solid_brt, dtype=float)
                solid_brts[nu] = solid_brt

            for i, (nu1, nu2) in enumerate(frequency_pairs):
                a, b = A[i], B[i]
                t_avg_up = T_avg_up[i]
                mat = M[i]
                tau_o = Tau_o[i]

                solid_brt = np.asarray([solid_brts[nu1], solid_brts[nu2]])
                solid_brt = np.moveaxis(solid_brt, 0, -1)

                D = b * b - 4 * a * (solid_brt - t_avg_up)
                tau_e = math.as_tensor(-math.log((-b + math.sqrt(D)) / (2 * a))) * np.cos(angle)

                right = math.move_axis(tau_e - tau_o, 0, -1)

                sol = math.linalg_solve(mat, right)
                Wrvs = np.asarray(sol[1, :], dtype=float)
                Qrvs = np.asarray(sol[0, :], dtype=float)

                Wr = Wrvs[0]
                Qr = Qrvs[0]

                Wrv = Wrvs[1:]
                Qrv = Qrvs[1:]

                DWrv = Wrv - Wr
                DQrv = Qrv - Qr

                dWrv = (Wrv - Wr) / Wr * 100.
                dQrv = (Qrv - Qr) / Qr * 100.

                data.append([
                    THETA, h_true, q_true, w_true,
                    i, nu1, nu2, Qr, Wr, Qrv, Wrv,
                    DQrv, DWrv, dQrv, dWrv,
                ])

        with open('mdpi.bin', 'wb') as dump:
            dill.dump(data, dump, recurse=True)

    with open('mdpi.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)
