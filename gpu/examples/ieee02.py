# -*- coding: utf-8 -*-
import numpy as np
# from matplotlib import pyplot as plt
import os
from collections import defaultdict
import dill
import datetime

from gpu.atmosphere import Atmosphere
from cpu.core.domain import Domain3D
from cpu.cloudiness import Plank3D, liquid_water, Cloudiness3D
from gpu.surface import SmoothWaterSurface
import gpu.satellite as satellite

from cpu.utils import map2d


if __name__ == '__main__':
    project_folder = 'ieee02'
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)

    H = 20.     # высота атмосферы
    d = 100     # дискретизация по высоте
    X = 50     # (км) горизонтальная протяженность моделируемой атмосферной ячейки
    N = 300     # горизонтальная дискретизация

    atmosphere = Atmosphere.Standard(H=H, dh=H / d)     # для облачной атмосферы по Планку
    solid = Atmosphere.Standard(H=H, dh=H / d)          # для атмосферы со сплошной облачностью

    atmosphere.integration_method = 'trapz'             # метод интегрирования
    solid.integration_method = atmosphere.integration_method

    # atmosphere.angle = 30. * np.pi / 180.             # зенитный угол наблюдения, по умолчанию: 0
    atmosphere.horizontal_extent = X
    atmosphere.incline = 'left'                         # в какую сторону по Ox наклонена траектория наблюдения
    solid.angle = atmosphere.angle
    solid.horizontal_extent = atmosphere.horizontal_extent
    solid.incline = atmosphere.incline

    surface = SmoothWaterSurface()                      # модель гладкой водной поверхности
    surface.angle = atmosphere.angle

    #########################################################################
    ID = -1
    freqs = [22.2, 36, 89]  # рассматриваемые частоты в ГГц
    # DIAP = np.linspace(1., 5., 20)  # диапазон значений Dm
    DIAP = np.arange(1., 5.1, 0.5)

    # kernel = (60, 60)   # 10 km - размер элемента разрешения в узлах
    # kernel = 30 # 5 km - размер элемента разрешения в узлах
    # kernels = [int(a) for a in np.arange(30, 90+1, 6)]
    kernels = [int(a) for a in np.arange(6, 294+1, 6)]

    wh_corr = 1.  # корректировочный коэффициент для зависимости W от H

    # K = 100
    K_range = np.arange(70, 214, 15)
    # beta = -0.9
    beta_range = np.arange(-0.9, 1.1, 0.2)

    alpha = 1.0
    eta = 1.0
    #########################################################################

    #########################################################################

    start_time = datetime.datetime.now()

    for K in K_range:

        for beta in beta_range:
            print("\n\nK is {}\t\tBETA is {:.2f}\n".format(K, beta))

            #####################################################################

            ID += 1
            datadict = {
                'id': ID,
                'params': {

                    'H': H,  # высота моделируемой атмосферы
                    'dh': d,  # дискретизация по высоте
                    'X': X,  # (км) горизонтальная протяженность моделируемой атмосферной ячейки
                    'N': N,  # горизонтальная дискретизация

                    'whc': wh_corr,  # корректировочный коэффициент для зависимости W от H

                    'kernels': [],  # размер элемента разрешения в узлах

                    'K': K,     # нормировочный коэффициент в модели Планка
                    'beta': beta,    # коэффициент beta из модели Планка
                    'alpha': alpha,
                    'eta': eta,

                    'dm_diap': [],  # диапазон значений Dm (список)

                    'freqs': freqs,   # рассматриваемые частоты в ГГц (список)

                },
                'w': {
                    'real_percentage': None,
                    'total_max': [],    # максимальный водозапас во всей атмосферной ячейке при данном Dm

                },
                'delta': {

                }     # разности яркостных температур

            }

            for kernel in kernels:
                datadict['w'].update({str(kernel): []})
                datadict['delta'].update({str(kernel): defaultdict(list)})

            for i, dm in enumerate(DIAP):
                print('\r{:.2f}%'.format(i / (len(DIAP) - 1) * 100), end='  ', flush=True)

                try:
                    atmosphere.liquid_water = Plank3D(kilometers=(X, X, H), nodes=(N, N, d)).liquid_water(
                        K=K, Dm=dm, beta=beta,
                        alpha=alpha, eta=eta,
                        _w=lambda _h: wh_corr * 0.132574 * np.power(_h, 2.30215),
                        verbose=False,
                        timeout=1,
                    )
                except TimeoutError:
                    print('\n...\n')
                    continue

                datadict['params']['dm_diap'].append(dm)

                brts = []
                for nu in freqs:
                    # start_time = time.time()
                    brt = satellite.brightness_temperature(nu, atmosphere, surface, cosmic=True)
                    # print("--- %s seconds ---" % (time.time() - start_time))
                    brt = np.asarray(brt, dtype=float)
                    brts.append(brt)

                W = atmosphere.W

                # fig = plt.figure()
                # plt.title(r'W, кг/м$^2$')
                # plt.xlabel('Ось X, узлы')
                # plt.ylabel('Ось Y, узлы')
                # plt.imshow(W)
                # plt.colorbar()
                # plt.savefig(os.path.join(project_folder, '{}.png'.format(i)), dpi=300)
                # plt.close(fig)

                datadict['w']['total_max'].append(np.max(W))

                sh, sw = W.shape
                datadict['w']['real_percentage'] = np.count_nonzero(W) / (sh * sw) * 100.

                for kernel in kernels:

                    elapsed = datetime.datetime.now() - start_time
                    days = elapsed.days
                    hours = elapsed.seconds // 3600
                    minutes = (elapsed.seconds - hours * 3600) // 60
                    seconds = elapsed.seconds - hours * 3600 - minutes * 60

                    print('\r{:.2f}%\t-\t({}, {})\t-\t{} d\t{} h\t{} m\t{} s'.format(i / (len(DIAP) - 1) * 100,
                                                                                     kernel, kernel,
                                                                                     days, hours, minutes, seconds),
                          end='  ', flush=True)

                    datadict['params']['kernels'].append(str(kernel))

                    # свертка карты водозапаса с элементом разрешения выбранного размера
                    conv_w = map2d.conv_averaging(W, kernel=kernel)

                    datadict['w'][str(kernel)].append(np.mean(conv_w))

                    # обратный переход от водозапаса к высотам с учетом сделанной ранее коррекции
                    conv_h = np.power(conv_w / (wh_corr * 0.132574), 1. / 2.30215)

                    # m = int(np.floor(np.sqrt(len(conv_h))))     # !!!
                    # conv_h = np.reshape(conv_h[:m*m], (m, m))   # !!!

                    solid.liquid_water = Cloudiness3D(kilometers=(N//X, N//X * len(conv_h), H),
                                                      nodes=(1, len(conv_h), d), clouds_bottom=1.5).liquid_water(
                        np.asarray([conv_h]), const_w=False,
                        _w=lambda _h: wh_corr * 0.132574 * np.power(_h, 2.30215),
                    )
                    # # !!!
                    # solid.liquid_water = Cloudiness3D(kilometers=(N//X * m, N//X * m, H),
                    #                                   nodes=(m, m, d), clouds_bottom=1.5).liquid_water(
                    #     np.asarray(conv_h), const_w=False,
                    #     _w=lambda _h: wh_corr * 0.132574 * np.power(_h, 2.30215),
                    # )

                    for j, nu in enumerate(freqs):

                        conv_brt = map2d.conv_averaging(brts[j], kernel=kernel)
                        # conv_brt = conv_brt[:m*m]   # !!!

                        # start_time = time.time()
                        solid_brt = satellite.brightness_temperature(nu, solid, surface, cosmic=True)
                        # print("--- %s seconds ---" % (time.time() - start_time))
                        solid_brt = np.asarray(solid_brt, dtype=float)
                        # solid_brt = solid_brt.ravel()   # !!!

                        delta = solid_brt - conv_brt
                        datadict['delta'][str(kernel)][nu].append([np.mean(delta),
                                                                   np.min(delta), np.max(delta),
                                                                   np.var(delta), np.std(delta),
                                                                   np.max(delta) - np.min(delta)])

            with open(os.path.join(project_folder, '{}.part'.format(str(ID).zfill(10))), 'wb') as dump:
                dill.dump(datadict, dump)

            # datadict.clear()
            #
            # with open(os.path.join(project_folder, '{}.part'.format(str(ID).zfill(10))), 'rb') as dump:
            #     datadict = dill.load(dump)
            #
            # fig, ax = plt.subplots()
            # ax.set_title(r'5$\times$5 km r.e.')
            # ax.set_xlabel(r'$D_m$, km')
            # ax.set_ylabel(r'$\Delta(\nu) = ' +
            #               r'\left[ T_b^*(\nu) - \left[ T_b(\nu)\right]^{\mathrm{r.e.}} \right]_{\mathrm{conv}}$, K')
            # colors = ['darkblue', 'forestgreen', 'crimson']
            # ls = ['-', '--', '-.']
            # lns = []
            # DIAP = datadict['params']['dm_diap']
            # DELTA = datadict['delta']['30']
            # for i, key in enumerate(DELTA.keys()):
            #     ln = ax.plot(DIAP, DELTA[key][:, 0],
            #                  label='({}) '.format(i+1) + r'$\nu$ = ' + '{:.1f}  GHz'.format(key),
            #                  color=colors[i], linestyle=ls[i])
            #     lns += ln
            #
            # ax = ax.twinx()
            # ax.set_ylabel(r'W, kg/m$^2$')
            # ln = ax.scatter(DIAP, datadict['w']['30'], label='(4) Mean LWC, r.e.', color='black', marker='^')
            # ln = ax.scatter(DIAP, datadict['w']['total_max'], label='(5) Max LWC, totally', color='black', marker='x')
            #
            # labs = [l.get_label() for l in lns]
            # ax.legend(lns, labs, loc='best', frameon=False)
            #
            # if not os.path.exists(os.path.join(project_folder, 'pic')):
            #     os.makedirs(os.path.join(project_folder, 'pic'))
            # plt.savefig(os.path.join(project_folder, 'pic', 'forward_{}.png'.format(str(ID).zfill(10))), dpi=300)
            # plt.show()
