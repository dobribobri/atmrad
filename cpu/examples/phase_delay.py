
import numpy as np
import cpu.core.math as math
import dill
from cpu.core.static.water.dielectric import epsilon_complex
# from matplotlib import pyplot as plt


if __name__ == '__main__':
    data = {}

    for f in np.arange(1., 100., 0.2):
        f = np.round(f, decimals=1)
        data[f] = {}

        for t in np.arange(-10, 30, 0.2):
            t = np.round(t, decimals=1)
            print('\r{:.1f} GHz, \t\t\tT = {:.1f} C'.format(f, t), end='   ', flush=True)

            ##########################
            eps_c = epsilon_complex(f, t, Sw=0., mode=1.)    # Rec. ITU-R 840
            # eps_c = epsilon_complex(f, t, Sw=0., mode=0.)   # Монография

            ##########################
            xi = (eps_c - 1) / (eps_c + 2)
            gamma = math.im( xi )
            phi = math.re( xi )
            ##########################

            data[f][t] = [eps_c, phi, gamma]

    with open('epsc.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)
