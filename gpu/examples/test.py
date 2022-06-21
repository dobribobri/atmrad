# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import time
import re

from gpu.atmosphere import Atmosphere
from cpu.cloudiness import Plank3D
from gpu.surface import SmoothWaterSurface
import gpu.satellite as satellite
from cpu.atmosphere import Atmosphere as cpuAtm
from cpu.atmosphere import avg
from cpu.weight_funcs import krho
from cpu.core.static.weight_funcs import kw
import gpu.core.math as math

from gpu.examples.old.domain import Domain


def test1():
    sa = Atmosphere.Standard()
    sa.angle = 10 * np.pi / 180.
    freqs = np.linspace(18.0, 27.2, 47)
    brts = np.asarray([sa.downward.brightness_temperature(f) for f in freqs])
    print(brts)


def ex1():
    Dm = 3.
    K = 100
    alpha = 1.
    seed = 42
    beta = -0.9
    eta = 1.
    d = 500
    atmosphere = Atmosphere.Standard(H=10., dh=10./d)
    atmosphere.liquid_water = Plank3D(nodes=(300, 300, d)).liquid_water(
        Dm=Dm, K=K, alpha=alpha, beta=beta, eta=eta, seed=seed, timeout=30., verbose=True
    )

    import dill
    with open('ex1_lw.bin', 'wb') as dump:
        dill.dump(atmosphere.liquid_water, dump)

    # atmosphere.effective_cloud_temperature = -2.
    atmosphere.integration_method = 'boole'

    atmosphere.angle = 30. * np.pi / 180.
    # atmosphere.horizontal_extent = 50.  # km
    atmosphere.incline = 'left'

    surface = SmoothWaterSurface()
    surface.angle = atmosphere.angle

    start_time = time.time()
    brt = satellite.brightness_temperature(37, atmosphere, surface, cosmic=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(brt.shape)
    print(brt.dtype)

    plt.figure('brightness temperature')
    plt.title('Brightness temperature (K) at 22.2 GHz')
    ticks_pos = np.asarray([30, 60, 90, 120, 150, 180, 210, 240, 270])
    ticks_labels = np.round(ticks_pos / 300. * 50, decimals=0)
    ticks_labels = [int(i) for i in ticks_labels]
    plt.xticks(ticks_pos, ticks_labels)
    plt.yticks(ticks_pos, ticks_labels)
    plt.xlabel('km')
    plt.ylabel('km')
    plt.imshow(np.asarray(brt, dtype=float))
    plt.colorbar()
    plt.savefig('ex1.png', dpi=300)
    plt.show()


def ex3():
    atmosphere = Atmosphere.Standard(H=10, dh=10/500)
    atmosphere.integration_method = 'boole'

    surface = SmoothWaterSurface()
    surface.temperature = 15.
    surface.salinity = 0.

    freqs, tbs = [], []
    with open('tbs_check.txt', 'r') as file:
        for line in file:
            line = re.split(r'[ \t]', re.sub(r'[\r\n]', '', line))
            f, tb = [float(n) for n in line if n]
            freqs.append(f)
            tbs.append(tb)

    # freqs_ = np.linspace(18., 27.2, 47)
    freqs_ = np.linspace(10, 150, 100)
    start_time = time.time()
    brt = [satellite.brightness_temperature(f, atmosphere, surface, cosmic=False) for f in freqs_]
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure('brightness temperature')
    plt.xlabel('frequency, GHz')
    plt.ylabel('brightness temperature, K')
    plt.ylim((50, 300))
    plt.scatter(freqs, tbs, label='test', marker='x', color='black')
    plt.plot(freqs_, np.asarray(brt, dtype=float), label='result')
    plt.legend(loc='best', frameon=False)
    plt.savefig('ex3.png', dpi=300)
    plt.show()


def test2():
    #########################################################################
    # domain parameters
    H = 20.  # высота атмосферы
    d = 100  # дискретизация по высоте
    X = 50  # (км) горизонтальная протяженность моделируемой атмосферной ячейки
    res = 300  # горизонтальная дискретизация

    # observation parameters
    angle = 0
    incline = 'left'
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

    kernels = [int(a) for a in np.arange(6, 294 + 1, 6)]
    #########################################################################

    atmosphere = Atmosphere.Standard(H=H, dh=H / d, T0=T0, P0=P0, rho0=rho0)  # для облачной атмосферы по Планку
    solid = Atmosphere.Standard(H=H, dh=H / d, T0=T0, P0=P0, rho0=rho0)  # для атмосферы со сплошной облачностью

    atmosphere.integration_method = integration_method  # метод интегрирования
    solid.integration_method = atmosphere.integration_method

    # atmosphere.angle = 30. * np.pi / 180.             # зенитный угол наблюдения, по умолчанию: 0
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
        {'name': 'L3', 'alpha': 1.485, 'Dm': 4.020, 'dm': 0.03048, 'eta': 0.76, 'beta': -0.3, 'cl_bottom': 1.3716},
        # {'name': 'L1', 'alpha': 3.853, 'Dm': 1.448, 'dm': 0.01524, 'eta': 0.98, 'beta': 0.0, 'cl_bottom': 0.54864},
        {'name': 'T7', 'alpha': 1.35, 'Dm': 3.733, 'dm': 0.04572, 'eta': 1.2, 'beta': 0.0, 'cl_bottom': 1.24968},
        {'name': 'T6', 'alpha': 1.398, 'Dm': 3.376, 'dm': 0.03048, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 1.0668},
        {'name': 'T8', 'alpha': 1.485, 'Dm': 4.02, 'dm': 0.06096, 'eta': 1.2, 'beta': 0.4, 'cl_bottom': 1.3716},
        # {'name': 'T5', 'alpha': 2.051, 'Dm': 2.574, 'dm': 0.02286, 'eta': 0.85, 'beta': -0.13, 'cl_bottom': 1.11252},
        # {'name': 'T3', 'alpha': 2.361, 'Dm': 2.092, 'dm': 0.01524, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 0.82296},
        {'name': 'T9', 'alpha': 2.485, 'Dm': 2.656, 'dm': 0.04572, 'eta': 1.3, 'beta': 0.3, 'cl_bottom': 1.40208},
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
    T_cosmic = 2.7
    for i, freq_pair in enumerate([(frequencies[0], frequencies[n]) for n in range(1, len(frequencies))]):
        k_rho = [krho(sa, f) for f in freq_pair]
        k_w = [kw(f, t=-2.) for f in freq_pair]
        m = math.as_tensor([k_rho, k_w])
        M[i] = math.transpose(m)

        T_avg_down[i] = math.as_tensor([avg.downward.T(sa, nu) for nu in freq_pair])
        T_avg_up[i] = math.as_tensor([avg.upward.T(sa, nu) for nu in freq_pair])
        Tau_o[i] = math.as_tensor([sa.opacity.oxygen(nu) for nu in freq_pair])
        R = math.as_tensor([surface.reflectivity(nu) for nu in freq_pair])
        kappa = 1 - R
        A[i] = (T_avg_down[i] - T_cosmic) * R
        B[i] = T_avg_up[i] - T_avg_down[i] * R - math.as_tensor(surface_temperature + 273.15) * kappa
    #########################################################################

    distr = distributions[0]
    alpha, Dm, dm, eta, beta, cl_bottom = \
        distr['alpha'], distr['Dm'], distr['dm'], distr['eta'], distr['beta'], distr['cl_bottom']

    xi = -np.exp(-alpha * Dm) * (((alpha * Dm) ** 2) / 2 + alpha * Dm + 1) + \
        np.exp(-alpha * dm) * (((alpha * dm) ** 2) / 2 + alpha * dm + 1)

    required_percentage = 0.5
    K = 2 * np.power(alpha, 3) * (X * X * required_percentage) / (np.pi * xi)

    p = Plank3D(kilometers=(X, X, H), nodes=(res, res, d), clouds_bottom=cl_bottom)
    print('Generating cloud distribution...')
    clouds = []
    try:
        clouds = p.generate_clouds(
            Dm=Dm, dm=dm, K=K, alpha=alpha, beta=beta, eta=eta, seed=seed, timeout=1., verbose=True
        )
    except TimeoutError:
        print('\n ...time is over')
        exit(-1)

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
    for nu in frequencies:
        brt = satellite.brightness_temperature(nu, atmosphere, surface, cosmic=True)
        # brt = atmosphere.downward.brightness_temperature(nu, background=True)     # OK

        brt = np.asarray(brt, dtype=float)
        brts[nu] = brt

    # plt.figure('nu222')
    # plt.imshow(brts[22.2])
    # plt.colorbar()
    # domain = Domain(kilometers=(50, 50, 10), nodes=(res, res, 100))
    # domain.apply_hmap(hmap)
    # tb1, tb2 = domain.get_tb2d_sat_parallel([22.2, 27.2], V_polarization=False)
    # tb1, _ = domain.get_tb2d_sat(22.2)
    # plt.figure('nu222d')
    # plt.imshow(tb1)
    # plt.colorbar()
    # plt.show()

    W = atmosphere.W

    plt.figure('W')
    plt.imshow(W)
    plt.colorbar()

    Q = atmosphere.Q
    Q = [[Q] * res] * res

    plt.figure('Q')
    plt.imshow(Q)
    plt.colorbar()

    for i, freq_pair in enumerate([(frequencies[0], frequencies[j]) for j in range(1, len(frequencies))]):
        if i > 0:
            break

        a, b = A[i], B[i]
        t_avg_up = T_avg_up[i]
        t_avg_down = T_avg_down[i]

        mat = M[i]
        print('mat ', math.shape(mat))

        tau_o = Tau_o[i]

        brt = np.moveaxis(np.asarray([brts[nu] for nu in freq_pair]), 0, -1)
        brt = math.as_tensor(np.reshape(brt, (res*res, 2)))

        D = b * b - 4 * a * (brt - t_avg_up)
        tau_e = math.as_tensor(-math.log( (-b + math.sqrt(D)) / (2 * a) ))
        # tau_e = math.log(t_avg_down - T_cosmic) - math.log(t_avg_down - brt)  # OK
        print(tau_e)

        right = math.move_axis(tau_e - tau_o, 0, -1)
        print('right ', math.shape(right))

        mat = math.linalg_solve(mat, right)

        wbrt = np.asarray(mat[1, :], dtype=float)
        wbrt = np.reshape(wbrt, (res, res))

        plt.figure('wbrt_{}'.format(i))
        plt.imshow(wbrt)
        plt.colorbar()

        plt.figure('qbrt_{}'.format(i))
        plt.imshow(np.asarray(mat[0, :], dtype=float).reshape((res, res)))
        plt.colorbar()

    plt.show()


if __name__ == '__main__':

    # successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node,
    # so returning NUMA node zero ->
    # for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done

    # test1()
    # ex1()
    # ex3()
    test2()
