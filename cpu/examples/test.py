import numpy as np
from matplotlib import pyplot as plt
import time
import re

from cpu.atmosphere import Atmosphere
from cpu.cloudiness import Plank3D, CloudinessColumn
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
    for i, H in enumerate([0.35, 1.5, 3]):

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
    plt.xscale('log')
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


if __name__ == '__main__':

    # test1()
    # ex1()
    ex3()
    # test2()
