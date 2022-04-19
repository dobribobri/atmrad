
# from cpu import ar
import os.path
import sys

from old.cpu import ar
from cpu import ar as ar_cpu
# from gpu import ar

import numpy as np
import time
import re
from termcolor import colored
import dill

from matplotlib import pyplot as plt

from multiprocessing import Process, Manager
from collections import Counter
from scipy.interpolate import splprep, splev
from scipy.optimize import curve_fit


def ex1():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.liquid_water = ar.Plank().get_lw_dist(K=100)
    atmosphere.integration_method = 'trapz'
    # dh = np.asarray([atmosphere.dh for _ in range(ar.op.len(atmosphere.temperature))])
    # atmosphere.dh = dh
    # atmosphere.angle = 0. * np.pi / 180.
    atmosphere.angle = 51. * np.pi / 180.
    atmosphere.horizontal_extent = 50.  # km
    # atmosphere.use_storage = False

    surface = ar.SmoothWaterSurface()
    surface.angle = atmosphere.angle

    start_time = time.time()
    brt = ar.satellite.multi.brightness_temperature([22.2], atmosphere, surface)
    print(atmosphere.storage.keys())
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.figure('brightness temperature')
    plt.xlabel('X, nodes')
    plt.ylabel('Y, nodes')
    # plt.imshow(block_averaging(np.asarray(brt[1], dtype=float), 50), cmap='Purples')
    plt.imshow(np.asarray(brt[0], dtype=float))
    plt.colorbar()
    plt.savefig('ex4.png', dpi=300)
    plt.show()


def ex2():
    atmosphere = ar.Atmosphere.Standard()
    atmosphere.attenuation.summary(22.2)
    print(atmosphere.storage.keys())

    atmosphere.liquid_water = ar.Plank().get_lw_dist(verbose=False)
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
    brt = [ar.satellite.brightness_temperature(f, atmosphere, surface) for f in freqs_]
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

    d = '04'
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
            s += colored('{:.2f}% '.format(sa.relative_humidity[i]), 'red') + colored('{:.2f}%\t'.format(rel[i]),
                                                                                      'green')
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


def ex6():
    a = np.ones((20, 25))
    b = ar.map.add_zeros(a, bounds=(3, 4))
    print(b)


def ex7():
    theta = 51. * np.pi / 180.
    PX = 50.
    PZ = 10.
    Nx = 300
    Nz = 500
    dh = PZ / Nz

    h_map = ar.Plank((PX, PX, PZ), (Nx, Nx, Nz)).h_map(K=100)

    shift = dh * PZ / np.cos(theta) * np.sin(theta)
    n = int(np.round(shift * len(h_map) / 2.))
    print('n = {}'.format(n))
    h_map = ar.map.add_zeros(h_map, bounds=(n, 0))

    atmosphere = ar.Atmosphere.Standard(H=PZ, dh=dh)
    atmosphere.integration_method = 'trapz'
    atmosphere.angle = theta
    atmosphere.horizontal_extent = PX + shift  # km
    atmosphere.use_storage = False
    atmosphere.liquid_water = ar.Plank((PX + shift, PX, PZ), (Nx + 2 * n, Nx, Nz)).lw_dist(h_map)

    brt = []
    surface = ar.SmoothWaterSurface(polarization='H')
    surface.angle = atmosphere.angle
    start_time = time.time()
    brt.append(np.asarray(ar.satellite.multi.brightness_temperature([22.2], atmosphere, surface)[0], dtype=float))
    print(atmosphere.storage.keys())
    print("--- %s seconds ---" % (time.time() - start_time))

    surface = ar.SmoothWaterSurface(polarization='V')
    surface.angle = atmosphere.angle
    start_time = time.time()
    brt.append(np.asarray(ar.satellite.multi.brightness_temperature([22.2], atmosphere, surface)[0], dtype=float))
    print(atmosphere.storage.keys())
    print("--- %s seconds ---" % (time.time() - start_time))

    vmin, vmax = np.min(brt), np.max(brt)
    print(vmin, vmax)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    title = ['H polarization', 'V polarization']
    for i, ax in enumerate(axes.flat):
        ax.set_title(title[i])
        ax.set_xlabel('X, nodes')
        ax.set_ylabel('Y, nodes')
        im = ax.imshow(brt[i], vmin=vmin, vmax=vmax)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, ax=axes.ravel().tolist(), cax=cbar_ax)
    plt.show()

    plt.figure('H polarization')
    # plt.title('H polarization')
    plt.xlabel('км')
    plt.ylabel('км')
    plt.xticks([0, 50, 100, 150, 200, 250], [int(a) for a in np.round(np.array([0, 50, 100, 150, 200, 250]) / 300 * 50, decimals=0)])
    plt.yticks([0, 50, 100, 150, 200, 250], [int(a) for a in np.round(np.array([0, 50, 100, 150, 200, 250]) / 300 * 50, decimals=0)])
    plt.imshow(brt[0], vmin=vmin, vmax=vmax, cmap='gray')
    plt.savefig('h.png', dpi=300)
    plt.colorbar()

    plt.figure('V polarization')
    # plt.title('V polarization')
    plt.xlabel('км')
    plt.ylabel('км')
    plt.xticks([0, 50, 100, 150, 200, 250], [int(a) for a in np.round(np.array([0, 50, 100, 150, 200, 250]) / 300 * 50, decimals=0)])
    plt.yticks([0, 50, 100, 150, 200, 250], [int(a) for a in np.round(np.array([0, 50, 100, 150, 200, 250]) / 300 * 50, decimals=0)])
    plt.imshow(brt[1], vmin=vmin, vmax=vmax, cmap='gray')
    plt.colorbar()
    plt.savefig('v.png', dpi=300)
    plt.show()

    print(np.array(brt[1]).shape)


def ex8():
    PX = 100.
    PZ = 10.
    Nx = 300
    Nz = 500

    n_workers = 8

    if not os.path.exists('hmaps'):
        os.makedirs('hmaps')
    for i in range(n_workers):
        if not os.path.exists(os.path.join('hmaps', '{}'.format(i))):
            os.makedirs(os.path.join('hmaps', '{}'.format(i)))

    def process(Dm, alpha, beta, eta, seed):
        valid = False
        for K in range(1, 1000, 100):
            try:
                h_map = ar.Plank((PX, PX, PZ), (Nx, Nx, Nz)).h_map(Dm, K, alpha, beta, eta, seed,
                                                                   timeout=1,
                                                                   verbose=False)
                if np.any(h_map):
                    valid = True
                    break
            except Exception:
                continue
        if not valid:
            return

        for K in range(1, 1000, 1):
            try:
                h_map = ar.Plank((PX, PX, PZ), (Nx, Nx, Nz)).h_map(Dm, K, alpha, beta, eta, seed,
                                                                   timeout=1,
                                                                   verbose=False)

                item = os.path.join('hmaps', '{}'.format(seed),
                       'Dm_{:.1f}__alpha_{:.2f}__beta_{:.2f}__eta_{:.2f}'.format(Dm, alpha, beta, eta))
                if not os.path.exists(item):
                    os.makedirs(item)

                with open(os.path.join(item,
                                       'K_{}'.format(
                                           K
                                       )), 'wb') as file:
                    dill.dump(np.asarray(h_map, dtype=np.float16), file, recurse=True)

            except Exception as e:
                print(colored('\nException: {}\n'.format(e), 'red'))
                print(colored('завершаю процесс {}...'.format(seed), 'green'))
                break
        return

    Dm_range = np.arange(1., 5., 1.)
    alpha_range = np.arange(0.25, 3., 0.25)
    beta_range = np.arange(0.25, 3., 0.25)
    eta_range = np.arange(0.25, 2., 0.25)
    N = len(Dm_range) * len(alpha_range) * len(beta_range) * len(eta_range)
    k = 0
    for Dm in Dm_range:
        for alpha in alpha_range:
            for beta in beta_range:
                for eta in eta_range:
                    k += 1
                    print(colored('=======================================================================', 'cyan'))
                    print(colored(
                         'Dm = {:.1f}\talpha = {:.2f}\tbeta = {:.2f}\teta = {:.2f}\t\t---\t\t{:.2f} %'.format(
                          Dm, alpha, beta, eta,  k / N * 100.
                          ), 'cyan'))
                    print(colored('=======================================================================', 'cyan'))

                    processes = []
                    for seed in range(8):

                        processes.append(Process(target=process, args=(Dm, alpha, beta, eta, seed,)))

                    ar.c.multi.do(processes, n_workers=n_workers)


def ex9():
    PX, PY, PZ = 50., 50., 10.
    Nx, Ny, Nz = 300, 300, 500
    K = 100
    Dm = 3
    alpha = 1
    beta = 0.5
    eta = 0.5
    clouds = ar.Plank((PX, PY, PZ), (Nx, Ny, Nz)).cloudiness(Dm, K, alpha, beta, eta, timeout=0.5)
    radii = [cloud.rx for cloud in clouds]
    heights = [cloud.height for cloud in clouds]
    cs = zip(heights, radii)
    c = Counter(cs)
    H, N, S = [], [], []
    for key in sorted(c.keys()):
        height, radius = key
        H.append(height)
        N.append(c[key])
        S.append(c[key] * np.pi * radius ** 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(r'$D_m$ = ' + '{:.1f} km,\t'.format(Dm) + r'$\alpha$ = ' + '{:.2f},\t'.format(alpha) +
                 r'$\beta$ = ' + '{:.2f},\t'.format(beta) + r'$\eta$ = ' + '{:.1f},\t'.format(eta) +
                 'K = {}'.format(K))
    ax.set_xlabel('H (heights), km')
    ax.set_ylabel('Number')

    ax.scatter(H, N, color='crimson', marker='+', label='Number of clouds $n(H)$')

    coeff, _ = curve_fit(lambda t, a, b, c: a + b*np.sqrt(t) + c*t,  H,  N)
    label = '{:.1f} '.format(coeff[0])
    if coeff[1] > 0:
        label += '+ {:.1f}'.format(np.abs(coeff[1]))
    else:
        label += '- {:.1f}'.format(np.abs(coeff[1]))
    label += r'$\cdot\sqrt{H}$ '
    if coeff[2] > 0:
        label += '+ {:.1f}'.format(np.abs(coeff[2]))
    else:
        label += '- {:.1f}'.format(np.abs(coeff[2]))
    label += r'$\cdot H$'
    ax.plot(H, [np.sum(coeff * np.array([1, np.sqrt(h), h])) for h in H], color='crimson', linestyle=':',
            label=label)

    ax.set_ylim((0, np.max(N) + 0.3 * np.max(N)))
    ax.legend(loc='upper right', frameon=False)

    ax2 = ax.twinx()
    ax2.set_ylim((0, np.max(S) + 0.3 * np.max(S)))
    ax2.set_ylabel('km$^2$')

    ax2.scatter(H, S, color='darkblue', marker='.', label='Space occupied')

    coeff, _ = curve_fit(lambda t, a, b, c: a + b * np.exp(-t) + c * np.exp(-2*t), H, S)
    label = '{:.1f} '.format(coeff[0])
    if coeff[1] > 0:
        label += ' + {:.1f}'.format(np.abs(coeff[1]))
    else:
        label += ' - {:.1f}'.format(np.abs(coeff[1]))
    label += r'$\cdot e^{-H}$ '
    if coeff[2] > 0:
        label += ' + {:.1f}'.format(np.abs(coeff[2]))
    else:
        label += ' - {:.1f}'.format(np.abs(coeff[2]))
    label += r'$\cdot e^{-2 H}$'
    ax2.plot(H, [np.sum(coeff * np.array([1, np.exp(-h), np.exp(-2*h)])) for h in H], color='darkblue', linestyle='--',
             label=label)
    ax2.legend(loc='upper left', frameon=False)

    full_square = np.sum(S)
    ns = []
    for i in range(len(S) - 1, -1, -1):
        ns.append(np.sum(S[i:]))
    ns = np.asarray(ns) / full_square * 100
    ns = np.flip(ns) - 90 > 0
    for i in range(len(ns)):
        if ns[i]:
            continue
        plt.fill_between(H[i:], [0] * len(H[i:]), [np.max(S) + 0.05 * np.max(S)] * len(H[i:]),
                         hatch='///////', facecolor=(0, 0, 0, 0), color='gray', alpha=0.2, linewidth=0)
        plt.text(np.mean(H[i:]), (np.max(S) + 0.15 * np.max(S)) / 2, '90% of $S^*$')
        break

    if not os.path.exists('nstat'):
        os.makedirs('nstat')

    path = os.path.join('nstat', 'K{}_Dm{:.1f}_alpha{:.2f}_beta{:.2f}_eta{:.1f}.png'.format(K, Dm, alpha, beta, eta))
    plt.savefig(path, dpi=300)
    plt.show()


def ex10():

    DATA = {}

    PX, PY = 50., 50.
    PZ = 10.
    Nx, Ny = 300, 300
    Nz = 500
    dh = PZ / Nz

    K = 200
    Dm = 3
    alpha = 1
    beta = -0.9

    freqs = [18.0, 22.2, 27.2, 37.5, 90.8]

    theta = 0.

    atmosphere = ar.Atmosphere.Standard(H=PZ, dh=dh)
    atmosphere.integration_method = 'trapz'
    atmosphere.use_storage = False
    atmosphere.angle = theta * np.pi / 180.

    surface_H = ar.SmoothWaterSurface(polarization='H')
    surface_V = ar.SmoothWaterSurface(polarization='V')
    surface_H.angle = theta * np.pi / 180.
    surface_V.angle = theta * np.pi / 180.

    for eta in [1.]:

        for seed in range(1):
            planck = ar.Plank((PX, PY, PZ), (Nx, Ny, Nz))
            h_map = planck.h_map(Dm, K, alpha, beta, eta, seed, timeout=0.3)
            atmosphere.liquid_water = planck.lw_dist(h_map)
            W = 0.132574 * np.power(h_map, 2.30215)

            AMOUNT = np.count_nonzero(h_map) / np.size(h_map) * 100

            start_time = time.time()
            BrTs_H = [np.asarray(a, dtype=float) for a in
                      ar.satellite.multi.brightness_temperature(freqs, atmosphere, surface_H)]
            print("--- %s seconds ---" % (time.time() - start_time))

            start_time = time.time()
            BrTs_V = [np.asarray(a, dtype=float) for a in
                      ar.satellite.multi.brightness_temperature(freqs, atmosphere, surface_V)]
            print("--- %s seconds ---" % (time.time() - start_time))

            for i, f in enumerate(freqs):
                for n in [2, 3, 4, 5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100, 150, 300]:
                    print(seed, f, n)
                    MLWC = ar.map.block_averaging(W, n, same_size=False).flatten()
                    # print(MLWC.shape)
                    MBRTH = ar.map.block_averaging(BrTs_H[i], n, same_size=False).flatten()
                    MBRTV = ar.map.block_averaging(BrTs_V[i], n, same_size=False).flatten()
                    # print(MBRTV.shape, MBRTH.shape)

                    H_MLWC = np.array([np.power(w / 0.132574, 1 / 2.30215) for w in MLWC])
                    # print(H_MLWC.shape)
                    h_map_mlwc = np.array([H_MLWC])
                    atmosphere.liquid_water = ar.Plank(nodes=(1, len(H_MLWC), 500)).lw_dist(h_map_mlwc)
                    BRTH_H_MLWC = ar.satellite.brightness_temperature(f, atmosphere, surface_H)[0]
                    BRTV_H_MLWC = ar.satellite.brightness_temperature(f, atmosphere, surface_V)[0]
                    # print(BRTH_H_MLWC.shape, BRTV_H_MLWC.shape)

                    DELTAH = (BRTH_H_MLWC - MBRTH) / MBRTH * 100.
                    DELTAV = (BRTV_H_MLWC - MBRTV) / MBRTV * 100.

                    DATA[(K, Dm, alpha, beta, eta, theta, seed, f, n)] = {'MLWC': (
                        np.mean(MLWC), np.min(MLWC), np.max(MLWC), np.var(MLWC)
                    ), 'MBRTH': (
                        np.mean(MBRTH), np.min(MBRTH), np.max(MBRTH), np.var(MBRTH)
                    ), 'MBRTV': (
                        np.mean(MBRTV), np.min(MBRTV), np.max(MBRTV), np.var(MBRTV)
                    ), 'H_MLWC': (
                        np.mean(H_MLWC), np.min(H_MLWC), np.max(H_MLWC), np.var(H_MLWC)
                    ), 'BRTH_H_MLWC': (
                        np.mean(BRTH_H_MLWC), np.min(BRTH_H_MLWC), np.max(BRTH_H_MLWC), np.var(BRTH_H_MLWC)
                    ), 'BRTV_H_MLWC': (
                        np.mean(BRTV_H_MLWC), np.min(BRTV_H_MLWC), np.max(BRTV_H_MLWC), np.var(BRTV_H_MLWC)
                    ), 'DELTA_MTBH': (
                        (np.mean(MBRTH) - np.mean(BRTH_H_MLWC)) / np.mean(MBRTH) * 100.,
                        (np.min(MBRTH) - np.min(BRTH_H_MLWC)) / np.min(MBRTH) * 100.,
                        (np.max(MBRTH) - np.max(BRTH_H_MLWC)) / np.max(MBRTH) * 100.,
                        (np.var(MBRTH) - np.var(BRTH_H_MLWC)) / np.var(MBRTH) * 100.
                    ), 'DELTA_MTBV': (
                        (np.mean(MBRTV) - np.mean(BRTV_H_MLWC)) / np.mean(MBRTV) * 100.,
                        (np.min(MBRTV) - np.min(BRTV_H_MLWC)) / np.min(MBRTV) * 100.,
                        (np.max(MBRTV) - np.max(BRTV_H_MLWC)) / np.max(MBRTV) * 100.,
                        (np.var(MBRTV) - np.var(BRTV_H_MLWC)) / np.var(MBRTV) * 100.
                    ), 'MDELTAH': (
                        np.mean(DELTAH), np.min(DELTAH), np.max(DELTAH), np.var(DELTAH)
                    ), 'MDELTAV': (
                        np.mean(DELTAV), np.min(DELTAV), np.max(DELTAV), np.var(DELTAV)
                    ), 'AMOUNT': AMOUNT
                    }

    with open('out_data.bin', 'wb') as dump:
        dill.dump(DATA, dump, recurse=True)


if __name__ == '__main__':

    ex3()
