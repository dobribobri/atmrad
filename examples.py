
# from cpu import ar
from gpu import ar
import numpy as np
import time
from matplotlib import pyplot as plt


def ex1():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.liquid_water = ar.Planck().get_lw_dist()

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


if __name__ == '__main__':

    ex1()
