
from typing import Union, Tuple
# from cpu import ar
from gpu import ar

import numpy as np
import time
import re
from termcolor import colored

from matplotlib import pyplot as plt


def __n(kernel: Union[Tuple, int]):
    if type(kernel) == int:
        ni = nj = kernel
    else:
        ni, nj = kernel
    return ni, nj


def block_averaging(array, kernel: Union[Tuple, int] = (10, 10), same_size=True) -> np.ndarray:
    ni, nj = __n(kernel)
    if same_size:
        for i in range(0, len(array), ni):
            for j in range(0, len(array[i]), nj):
                array[i:i + ni, j:j + nj] = np.mean(array[i:i + ni, j:j + nj])
    else:
        new_arr = np.zeros((len(array) // ni, len(array[0]) // nj), dtype=float)
        for i in range(0, len(array), ni):
            for j in range(0, len(array[i]), nj):
                new_arr[i // ni, j // nj] = np.mean(array[i:i + ni, j:j + nj])
        array = new_arr
    return array


def ex1():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.liquid_water = ar.Planck().get_lw_dist(K=100)
    atmosphere.integration_method = 'trapz'
    dh = np.asarray([atmosphere.dh for _ in range(ar.op.len(atmosphere.temperature))])
    atmosphere.dh = dh
    atmosphere.angle = 0. * np.pi / 180.
    # atmosphere.angle = 51. * np.pi / 180.
    # atmosphere.use_storage = False

    surface = ar.SmoothWaterSurface()

    start_time = time.time()
    brt = ar.satellite.multi.brightness_temperature([18.0, 22.2], atmosphere, surface)
    print(atmosphere.storage.keys())
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure('brightness temperature')
    plt.xlabel('X, nodes')
    plt.ylabel('Y, nodes')
    # plt.imshow(block_averaging(np.asarray(brt[1], dtype=float), 50), cmap='Purples')
    plt.imshow(np.asarray(brt[1], dtype=float))
    plt.colorbar()
    plt.savefig('ex4.png', dpi=300)
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

    frequencies = np.linspace(18.0, 27.2, 47)
    plt.figure('Нисходящая Я.Т.')
    plt.xlabel(r'Частота $\nu$, ГГц')
    plt.ylabel('Яркостная температура, K')

    d = '08'
    labels = ['По данным радиозонда от {}.10.2021 00Z'.format(d),
              'По данным радиозонда от {}.10.2021 12Z'.format(d)]
    colors = ['darkblue', 'darkorange']
    # tbd_std_avg = np.zeros_like(frequencies)
    for k, fname in enumerate(['Dolgoprudnyj_{}_10_00Z.txt'.format(d), 'Dolgoprudnyj_{}_10_12Z.txt'.format(d)]):
        T, P, rel, alt = [], [], [], []
        with open(fname, 'r') as file:
            for line in file:
                num = [float(e) for e in re.split(r'[ \t]', re.sub(r'[^0-9.\- \t]', '', line)) if e]
                valid = len(num) == 11
                if valid:
                    # print(num)
                    P.append(num[0])
                    alt.append(num[1] / 1000)
                    T.append(num[2])
                    rel.append(num[4])
        T, P, rel, alt = [ar.op.as_tensor(a) for a in [T, P, rel, alt]]
        sa = ar.Atmosphere.Standard(H=alt[-1], dh=alt[-1] / ar.op.len(T))
        print('==================================================================================================')
        print(colored('Стандарт ', 'red') + colored('Данные', 'green'))
        for i in range(ar.op.len(T)):
            s = colored('{:.2f} '.format(sa.temperature[i]), 'red') + colored('{:.2f}\t'.format(T[i]), 'green')
            s += colored('{:.2f} '.format(sa.pressure[i]), 'red') + colored('{:.2f}\t'.format(P[i]), 'green')
            s += colored('{:.2f}% '.format(sa.relative_humidity[i]), 'red') + colored('{:.2f}%\t'.format(rel[i]), 'green')
            print(s)
        print('==================================================================================================')
        print()

        atmosphere = ar.Atmosphere(T, P, RelativeHumidity=rel, altitudes=alt)
        atmosphere.integration_method = 'boole'
        tb_down = atmosphere.downward.brightness_temperatures(frequencies, n_workers=8)
        rho = atmosphere.absolute_humidity

        print('==================================================================================================')
        for i in range(ar.op.len(T)):
            s = colored('{:.2f} '.format(sa.absolute_humidity[i]), 'red') + colored('{:.2f}\t'.format(rho[i]), 'green')
            print(s)
        print('==================================================================================================')
        print()
        Q = ar.c.integrate.full(rho, atmosphere.dh, method='boole') / 10
        # plt.plot(frequencies, tb_down, label=labels[k], color=colors[k])
        plt.plot(frequencies, tb_down, label='{}, Q = {:.2f} '.format(labels[k], Q) + r'г/см$^2$', color=colors[k])

        # tb_down = sa.downward.brightness_temperatures(frequencies, n_workers=8)
        # tbd_std_avg = tbd_std_avg + tb_down

    # tbd_std_avg /= len(labels)
    # sa = ar.Atmosphere.Standard(H=30, dh=30. / 1000)
    # tbd_std_avg = sa.downward.brightness_temperatures(frequencies, n_workers=8)
    # rho = sa.absolute_humidity
    # Q = ar.c.integrate.full(rho, 30. / 1000, 'boole') / 10
    # plt.plot(frequencies, tbd_std_avg, label='Стандарт атмосферы, 30 км; Q = {:.2f} '.format(Q) + r'г/см$^2$',
    #          color='black', linestyle='--')

    plt.legend(loc='best', frameon=False)
    plt.savefig('{}.10.2021.png'.format(d), dpi=300)
    plt.show()


if __name__ == '__main__':

    ex1()
