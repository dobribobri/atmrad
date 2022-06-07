# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import time
import re

from gpu.atmosphere import Atmosphere
from cpu.cloudiness import Plank3D
from gpu.surface import SmoothWaterSurface
import gpu.satellite as satellite


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


if __name__ == '__main__':

    # successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node,
    # so returning NUMA node zero ->
    # for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done

    # test1()
    ex1()
    # ex3()
