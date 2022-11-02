
import numpy as np
import cpu.core.math as math
import dill
from cpu.core.static.water.dielectric import epsilon_complex
# from matplotlib import pyplot as plt


if __name__ == '__main__':
    data = {}

    for f in np.arange(1., 1000., 0.2):
        f = np.round(f, decimals=1)
        data[f] = {}

        for t in np.arange(-10, 30, 0.2):
            t = np.round(t, decimals=1)
            data[f][t] = {}

            print('\r{:.1f} GHz, \t\t\tT = {:.1f} C'.format(f, t), end='   ', flush=True)

            ##########################
            eps_c_840 = epsilon_complex(f, t, Sw=0., mode=1.)    # Rec. ITU-R 840
            eps_c_0 = epsilon_complex(f, t, Sw=0., mode=0.)   # Монография

            ##########################
            xi_840 = (eps_c_840 - 1) / (eps_c_840 + 2)
            gamma_840 = math.im( xi_840 )
            phi_840 = math.re( xi_840 )

            xi_0 = (eps_c_0 - 1) / (eps_c_0 + 2)
            gamma_0 = math.im( xi_0 )
            phi_0 = math.re( xi_0 )

            ##########################

            data[f][t]['ITU840'] = [eps_c_840, phi_840, gamma_840]
            data[f][t]['original'] = [eps_c_0, phi_0, gamma_0]

    with open('epsc.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)
