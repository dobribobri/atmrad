
# from cpu import ar
from gpu import ar

import numpy as np
import time
import re

from matplotlib import pyplot as plt


def ex1():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.liquid_water = ar.Planck().get_lw_dist()
    # atmosphere.use_storage = False

    surface = ar.SmoothWaterSurface()

    start_time = time.time()
    brt = ar.satellite.multi.brightness_temperature([18.0, 22.2], atmosphere, surface)
    print(atmosphere.storage.keys())
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure('brightness temperature')
    plt.imshow(np.asarray(brt[1], dtype=float))
    plt.colorbar()
    plt.savefig('ex1.png', dpi=300)
    plt.show()


def ex2():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.attenuation.summary(22.2)
    print(atmosphere.storage.keys())

    atmosphere.liquid_water = ar.Planck().get_lw_dist(verbose=False)
    print(atmosphere.storage.keys())


def ex3():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.effective_cloud_temperature = -2.
    atmosphere.integration_method = 'simpson'
    atmosphere.use_storage = False

    surface = ar.SmoothWaterSurface()
    surface.temperature = 15.
    surface.salinity = 0.

    freqs, tbs = [], []
    with open('tbs_check.txt', 'r') as file:
        for line in file:
            line = re.split(r'[ \t]', re.sub(r'[\r\n]', '', line))
            f, tb = [float(n) for n in line if n]
            freqs.append(f)
            tbs.append(tb)

    freqs_ = np.linspace(10., 150., 100)
    start_time = time.time()
    brt = ar.satellite.multi.brightness_temperature(freqs_, atmosphere, surface)
    # brt = [ar.satellite.brightness_temperature(f, atmosphere, surface) for f in freqs_]
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure('brightness temperature')
    plt.xlabel('frequency, GHz')
    plt.ylabel('brightness temperature, K')
    plt.ylim((50, 300))
    plt.scatter(freqs, tbs, label='test', marker='x', color='black')
    plt.plot(freqs_, np.asarray(brt, dtype=float), label='result')
    plt.legend(loc='best', frameon=False)
    plt.savefig('tbs_check.png', dpi=300)
    plt.show()


def ex4():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.effective_cloud_temperature = -2.
    atmosphere.integration_method = 'simpson'

    # print(ar.static.p676.gamma_oxygen(37.5, atmosphere.temperature, atmosphere.pressure))
    # print(ar.static.attenuation.oxygen(37.5, atmosphere.temperature, atmosphere.pressure))
    print(atmosphere.attenuation.oxygen(37.5))
    # print(ar.Atmosphere.attenuation.oxygen(atmosphere, 37.5))


def ex5():
    pass


if __name__ == '__main__':

    ex1()
