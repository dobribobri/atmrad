
import numpy as np
import dill
from cpu.core.static.weight_funcs import kw


if __name__ == '__main__':
    data = {}

    for f in np.round(np.arange(1., 300., 0.2), decimals=1):
        data[f] = {}

        for t in np.round(np.arange(-10, 30, 0.2), decimals=1):

            print('\r{:.1f} GHz, \t\t\tT = {:.1f} C'.format(f, t), end='   ', flush=True)

            ##########################
            kw_1dim_o = kw(f, t, mode='one-dimensional')
            kw_1dim_o_wrong = kw(f, t, mode='one-dimensional-wrong')
            kw_1dim_s = kw(f, t, mode='one-dimensional-simplified')
            kw_2dim_c = kw(f, t, mode='two-dimensional-c')
            kw_2dim_c_wrong = kw(f, t, mode='two-dimensional-c-wrong')
            kw_2dim_b = kw(f, t, mode='two-dimensional-b')
            kw_2dim_R = kw(f, t, mode='840-8')

            ##########################
            data[f][t] = [kw_1dim_o, kw_1dim_s, kw_2dim_c, kw_2dim_b, kw_2dim_R, kw_1dim_o_wrong, kw_2dim_c_wrong]

    with open('kwcheck.bin', 'wb') as dump:
        dill.dump(data, dump, recurse=True)
