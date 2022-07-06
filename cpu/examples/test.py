import numpy as np
from matplotlib import pyplot as plt
import time
import re
from collections import defaultdict

from cpu.atmosphere import Atmosphere
from cpu.cloudiness import Plank3D, CloudinessColumn, Cloudiness3D
from cpu.surface import SmoothWaterSurface
import cpu.satellite as satellite


def test1():
    sa = Atmosphere.Standard()
    sa.angle = 10 * np.pi / 180.
    freqs = np.linspace(18.0, 27.2, 47)
    brts = np.asarray([sa.downward.brightness_temperature(f) for f in freqs])
    print(brts)
    brts = sa.downward.brightness_temperatures(freqs, background=True)
    print(brts)


def test2():
    plt.figure()

    tbs = []
    sa = Atmosphere.Standard(H=20, dh=20. / 500)
    sa.integration_method = 'boole'
    sa.horizontal_extent = 1.  # km

    frequencies = np.linspace(10, 300., 500)

    linestyles = ['-', '-.', '--']
    for i, H in enumerate([0.0, 1.5, 3]):

        W = 0.132574 * np.power(H, 2.30215)
        sa.liquid_water = CloudinessColumn(kilometers_z=20., nodes_z=500, clouds_bottom=1.5).liquid_water(
            H, const_w=False,
        )
        # tb = satellite.brightness_temperatures(frequencies, sa, SmoothWaterSurface(polarization='H'),
        #                                        cosmic=True, n_workers=8)
        tb = sa.downward.brightness_temperatures(frequencies, background=True, n_workers=8)

        plt.plot(frequencies, tb[:, 0, 0],
                 label='({}) W = {:.2f} kg/m'.format(i+1, W) + r'$^2$',
                 linestyle=linestyles[i], color='black')
        print('\rH = {} km ready'.format(H), end='  ', flush=True)

    plt.xlabel(r'Frequency $\nu$, GHz')
    plt.ylabel(r'Brightness temperature, К')
    # plt.xscale('log')
    xticks = [10, 20, 30, 50, 90, 183, 300]
    plt.xticks(ticks=xticks, labels=xticks)
    plt.legend(frameon=False)
    plt.savefig('img01.eps', dpi=300)
    plt.show()


def ex1():
    d = 500
    atmosphere = Atmosphere.Standard(H=10., dh=10./d)
    atmosphere.liquid_water = Plank3D(nodes=(300, 300, d)).liquid_water(K=100)
    atmosphere.integration_method = 'boole'

    # atmosphere.angle = 30. * np.pi / 180.
    atmosphere.horizontal_extent = 50.  # km
    atmosphere.incline = 'left'

    surface = SmoothWaterSurface()
    surface.angle = atmosphere.angle

    start_time = time.time()
    brt = satellite.brightness_temperature(22.2, atmosphere, surface, cosmic=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(brt.shape)
    print(brt.dtype)

    plt.figure('brightness temperature')
    plt.xlabel('X, nodes')
    plt.ylabel('Y, nodes')

    plt.imshow(np.asarray(brt.T, dtype=float))
    plt.colorbar()
    plt.savefig('ex1.png', dpi=300)
    plt.show()


def ex3():
    atmosphere = Atmosphere.Standard(H=20., dh=20./100)
    atmosphere.effective_cloud_temperature = -2.
    atmosphere.integration_method = 'simpson'

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
    brt = satellite.brightness_temperatures(freqs_, atmosphere, surface, cosmic=False)
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


def ex4():

    # domain parameters
    H = 20.  # высота атмосферы
    d = 100  # дискретизация по высоте

    # observation parameters
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

    ###
    _c0 = 0.132574
    _c1 = 2.30215

    cl_bottom = 1.5
    ###

    w = np.asarray([0, 2, 5])

    ###

    solid = Atmosphere.Standard(H=H, dh=H / d, T0=T0, P0=P0, rho0=rho0)  # для атмосферы со сплошной облачностью
    solid.integration_method = integration_method

    surface = SmoothWaterSurface(temperature=surface_temperature,
                                 salinity=surface_salinity,
                                 polarization=polarization)  # модель гладкой водной поверхности

    h = np.power(w / _c0, 1. / _c1)

    solid.liquid_water = Cloudiness3D(kilometers=(1, len(h), H),
                                      nodes=(1, len(h), d), clouds_bottom=cl_bottom).liquid_water(
        np.asarray([h]), const_w=False, _w=lambda _H: _c0 * np.power(_H, _c1)
    )

    brts = {}
    for nu in frequencies:
        brts[nu] = []
        for THETA in np.linspace(0, 51, 10):
            angle = THETA * np.pi / 180.  # зенитный угол наблюдения, по умолчанию: 0
            surface.angle = angle
            brt = satellite.brightness_temperature(nu, solid, surface, cosmic=True, __theta=angle)[0]
            brts[nu].append(brt)
        brts[nu] = np.asarray(brts[nu])

    plt.figure()
    plt.plot(np.linspace(0, 51, 10), brts[22.2][:, 0], label=r'22.2 GHz, 0 kg$\cdot$m$^{-2}$', linestyle='-')
    plt.plot(np.linspace(0, 51, 10), brts[22.2][:, 1], label=r'22.2 GHz, 2 kg$\cdot$m$^{-2}$', linestyle='-')
    # plt.plot(np.linspace(0, 51, 10), brts[22.2][:, 2], label=r'22.2 GHz, 5 kg$\cdot$m$^{-2}$', linestyle='-')

    plt.plot(np.linspace(0, 51, 10), brts[36][:, 0], label=r'27.2 GHz, 0 kg$\cdot$m$^{-2}$', linestyle='--')
    plt.plot(np.linspace(0, 51, 10), brts[36][:, 1], label=r'27.2 GHz, 2 kg$\cdot$m$^{-2}$', linestyle='--')
    plt.legend(frameon=False)
    plt.show()


if __name__ == '__main__':

    # test1()
    # ex1()
    # ex3()
    # test2()
    ex4()
