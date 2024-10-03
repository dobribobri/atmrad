import numpy as np
from matplotlib import pyplot as plt
import time
import re
from collections import defaultdict

from cpu.atmosphere import Atmosphere
from cpu.cloudiness import Plank3D, CloudinessColumn, Cloudiness3D
from cpu.surface import SmoothWaterSurface
import cpu.satellite as satellite
from cpu.weight_funcs import Staelin


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

    _H, _d = 20., 100
    sa = Atmosphere.Standard(H=_H, dh=_H / _d)
    sa.integration_method = 'trapz'
    sa.horizontal_extent = 1.  # km
    sa.approx = False
    sa.effective_cloud_temperature = -2.

    frequencies = np.linspace(10, 300., 500)

    theta = None
    # theta = 30. * np.pi / 180
    srf = SmoothWaterSurface(polarization='H')
    # srf.angle = theta

    linestyles = ['-', '-.', '--']
    w = [2, 5, 6]
    __const, __tb = [], []
    for i, W in enumerate(w):

        _c0 = 0.132574
        _c1 = 2.30215
        H = np.power(W / _c0, 1. / _c1)
        # sa.liquid_water = CloudinessColumn(kilometers_z=_H, nodes_z=_d, clouds_bottom=1.5).liquid_water(
        #     H, const_w=True,
        # )
        # _const = satellite.brightness_temperatures(frequencies, sa, srf,
        #                                        cosmic=True, n_workers=8, __theta=theta)
        # __const.append(_const[:, 0, 0])

        # sa.liquid_water = CloudinessColumn(kilometers_z=_H, nodes_z=_d, clouds_bottom=1.5).liquid_water(
        #     H, const_w=False,
        # )
        start = time.time()
        tb = satellite.brightness_temperatures(frequencies, sa, srf,
                                               cosmic=True, n_workers=8, __theta=theta)
        stop = time.time()
        print(stop - start)
        # tb = sa.downward.brightness_temperatures(frequencies, background=False, n_workers=8)
        __tb.append(tb[:, 0, 0])

        plt.plot(frequencies, tb[:, 0, 0],
                 label='({}) W = {:.2f} kg/m'.format(i+1, W) + r'$^2$',
                 linestyle=linestyles[i], color='black')
        print('\rH = {} km ready'.format(H), end='  ', flush=True)

    import dill
    with open('mazin_vs_const_10-300GHz_H20km_d500_trapz_cl.bottom1.5km_eff.cl.temp-2.data', 'wb') as dump:
        dill.dump((frequencies, __const, __tb, w), dump)

    plt.xlabel(r'Frequency $\nu$, GHz')
    plt.ylabel(r'Brightness temperature, К')
    plt.xscale('log')
    xticks = [10, 20, 30, 50, 90, 183, 300]
    plt.xticks(ticks=xticks, labels=xticks)
    plt.legend(frameon=False)
    plt.savefig('mazin_H20km_trapz_eff.cl.temp-2.png', dpi=300)
    # plt.savefig('wconst_H20km.eps')
    plt.show()


def test21():
    plt.figure()

    _H, _d = 20., 500
    sa = Atmosphere.Standard(H=_H, dh=_H / _d)
    sa.integration_method = 'boole'
    sa.horizontal_extent = 1.  # km
    sa.approx = False
    # sa.effective_cloud_temperature = -2.

    frequencies = np.linspace(10, 300., 500)

    theta = None
    # theta = 30. * np.pi / 180
    srf = SmoothWaterSurface(polarization='H')
    # srf.angle = theta

    linestyles = ['-', '-.', '--', ':']
    # w = [2, 5, 6]
    h = [0, 1, 2, 3]
    __const, __tb = [], []
    # for i, W in enumerate(w):
    for i, H in enumerate(h):

        _c0 = 0.132574
        _c1 = 2.30215
        # H = np.power(W / _c0, 1. / _c1)
        W = _c0 * np.power(H, _c1)
        # sa.liquid_water = CloudinessColumn(kilometers_z=_H, nodes_z=_d, clouds_bottom=1.5).liquid_water(
        #     H, const_w=True,
        # )
        # _const = satellite.brightness_temperatures(frequencies, sa, srf,
        #                                        cosmic=True, n_workers=8, __theta=theta)
        # __const.append(_const[:, 0, 0])

        sa.liquid_water = CloudinessColumn(kilometers_z=_H, nodes_z=_d, clouds_bottom=1.1).liquid_water(
            H, const_w=False,
        )
        start = time.time()
        # tb = sa.downward.brightness_temperatures(frequencies, n_workers=8)
        tb = satellite.brightness_temperatures(frequencies, sa, srf, cosmic=True, )
        stop = time.time()
        print(stop - start)
        # tb = sa.downward.brightness_temperatures(frequencies, background=False, n_workers=8)
        __tb.append(tb[:, 0, 0])

        plt.plot(frequencies, tb[:, 0, 0],
                 label='({}) W = {:.2f} кг/м'.format(i+1, W) + r'$^2$',
                 linestyle=linestyles[i], color='black')
        print('\rH = {} km ready'.format(H), end='  ', flush=True)

    # import dill
    # with open('mazin_vs_const_10-300GHz_H20km_d500_trapz_cl.bottom1.5km_eff.cl.temp-2.data', 'wb') as dump:
    #     dill.dump((frequencies, __const, __tb, w), dump)

    import dill
    with open('tb_spectra.dump', 'wb') as dump:
        dill.dump(__tb, dump)

    plt.xlabel(r'Частота $\nu$, ГГц')
    plt.ylabel(r'Яркостная температура, К')
    plt.xscale('log')
    xticks = [10, 20, 30, 50, 90, 183, 300]
    plt.xticks(ticks=xticks, labels=xticks)
    plt.legend(frameon=False)
    plt.savefig('114.png', dpi=300)
    # plt.savefig('wconst_H20km.eps')
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


def ex4():

    # domain parameters
    H = 20.  # высота атмосферы
    d = 500  # дискретизация по высоте

    # observation parameters
    integration_method = 'trapz'

    # atmosphere parameters
    T0 = 15.
    P0 = 1013
    rho0 = 7.5

    # surface parameters
    surface_temperature = 15.
    surface_salinity = 0.

    # radiation parameters
    polarization = 'H'
    frequencies = [22.2, 27.2, 36, 89]
    approx = False

    ###
    _c0 = 0.132574
    _c1 = 2.30215

    cl_bottom = 1.1
    ###

    w = np.asarray([0, 1, 3])

    ###

    solid = Atmosphere.Standard(H=H, dh=H / d, T0=T0, P0=P0, rho0=rho0)  # для атмосферы со сплошной облачностью
    solid.integration_method = integration_method
    solid.approx = approx
    # solid.effective_cloud_temperature = -2.

    surface = SmoothWaterSurface(temperature=surface_temperature,
                                 salinity=surface_salinity,
                                 polarization=polarization)  # модель гладкой водной поверхности

    h = np.power(w / _c0, 1. / _c1)

    solid.liquid_water = Cloudiness3D(kilometers=(1, len(h), H),
                                      nodes=(1, len(h), d), clouds_bottom=cl_bottom).liquid_water(
        np.asarray([h]), const_w=False, _w=lambda _H: _c0 * np.power(_H, _c1)
    )

    brts = {}
    _THETA = np.linspace(0, 51, 20)
    for nu in frequencies:
        brts[nu] = []
        for THETA in _THETA:
            angle = THETA * np.pi / 180.  # зенитный угол наблюдения, по умолчанию: 0
            surface.angle = angle
            brt = satellite.brightness_temperature(nu, solid, surface, cosmic=True, __theta=angle)[0]
            brts[nu].append(brt)
        brts[nu] = np.asarray(brts[nu])

    import dill
    with open('flat_tb_36GHz_theta_noapprox_polarization{}.data'.format(polarization), 'wb') as dump:
        dill.dump((frequencies, _THETA, brts), dump)

    nu = 36
    plt.figure()
    plt.plot(_THETA, brts[nu][:, 0], label=r'$W = 0$ кг$\cdot$м$^{-2}$', linestyle='-')
    plt.plot(_THETA, brts[nu][:, 1], label=r'$W = 1$ кг$\cdot$м$^{-2}$', linestyle='--')
    plt.plot(_THETA, brts[nu][:, 2], label=r'$W = 3$ кг$\cdot$м$^{-2}$', linestyle='-.')
    # plt.plot(_THETA, brts[nu][:, 3], label='{:.1f}'.format(nu) + r' GHz, 8 kg$\cdot$m$^{-2}$', linestyle=':')

    # plt.plot(np.linspace(0, 51, 10), brts[36][:, 0], label=r'36 GHz, 0 kg$\cdot$m$^{-2}$', linestyle='-')
    # plt.plot(np.linspace(0, 51, 10), brts[36][:, 1], label=r'36 GHz, 2 kg$\cdot$m$^{-2}$', linestyle='--')
    # plt.plot(np.linspace(0, 51, 10), brts[36][:, 2], label=r'36 GHz, 5 kg$\cdot$m$^{-2}$', linestyle='-.')
    plt.legend(frameon=False)
    plt.show()


def test3():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams.update({'font.family': 'serif'})
    matplotlib.rcParams.update({'text.latex.preamble':
                                    r'\usepackage[russian]{babel}\usepackage{amsmath}'})

    plt.figure(figsize=(7, 8))

    _H, _d = 30., 1000
    sa25 = Atmosphere.Standard(H=_H, dh=_H / _d)
    plt.plot(sa25.temperature, np.linspace(0, _H, _d))
    sa5 = Atmosphere.Standard(H=_H, dh=_H / _d)
    plt.plot(sa5.temperature, np.linspace(0, _H, _d))
    sa25.temperature = sa25.temperature + sa25.temperature * 0.3
    sa5.temperature = sa5.temperature - sa5.temperature * 0.3
    # plt.plot(sa25.temperature, np.linspace(0, _H, _d))
    # plt.plot(sa5.temperature, np.linspace(0, _H, _d))
    plt.show()

    plt.figure()
    sa25.integration_method = 'trapz'
    sa25.horizontal_extent = 1.  # km
    sa25.approx = False
    sa5.integration_method = 'trapz'
    sa5.horizontal_extent = 1.  # km
    sa5.approx = False

    from scipy.interpolate import splrep, BSpline

    colors = ['darkblue', (1, 100/255, 0), (60/255, 170/255, 60/255), (0, 157/255, 146/255), 'black', 'indigo', 'crimson']
    ls = ['--', '--', '-.', '-', '-', '-', '-.']

    from scipy.signal import savgol_filter

    for i, nu in enumerate([20, 20.4, 21.2, 21.6, 22.235, 22.8, 23.6]):
        s25 = Staelin(sa25, nu)
        s5 = Staelin(sa5, nu)
        h = np.linspace(0, _H, _d)

        # tck25 = splrep(h, s25, s=0.001)
        # tck5 = splrep(h, s5, s=0.0001)
        # s25_2 = BSpline(*tck25)(h)
        # s5_2 = BSpline(*tck5)(h)
        s25 = s25 / np.max(s25)
        s5 = s5 / np.max(s5)
        # s25_2 = s25_2 / np.max(s25_2)
        # s5_2 = s5_2 / np.max(s5_2)
        # s25 = np.where((h < 5) | (h > 17), s25, s25_2)
        # s5 = np.where((h < 5) | (h > 17), s5, s5_2)
        # s25 = np.where((h < 10.5) | (h > 11.5), s25, 0)
        # s5 = np.where((h < 10.5) | (h > 11.5), s5, 0)

        # if nu == 22.235:
        #     plt.plot(s25, h, label='{:.3f} ГГц'.format(nu), color=colors[i], linewidth=2)
        # else:
        #     plt.plot(s25, h, label='{:.1f} ГГц'.format(nu), color=colors[i])

        if nu == 22.235:
            plt.fill_betweenx(h, savgol_filter(s5, 100, 3), savgol_filter(s25, 100, 3), color=colors[i], linewidth=2, alpha=0.8,
                              label='{:.3f} ГГц'.format(nu), zorder=9999)
        else:
            # plt.plot(savgol_filter(s25, 50, 5), h, label='{:.1f} ГГц'.format(nu), color=colors[i])
            plt.fill_betweenx(h, savgol_filter(s5, 100, 3), savgol_filter(s25, 100, 3), color=colors[i], alpha=0.8,
                              label='{:.3f} ГГц'.format(nu))

        # if nu == 22.235:
        #     plt.plot(s5, h, label='{:.3f} ГГц'.format(nu), color=colors[i], linewidth=2)
        # else:
        #     plt.plot(s5, h, label='{:.1f} ГГц'.format(nu), color=colors[i])

    plt.legend(frameon=False, bbox_to_anchor=(1.5, 1), borderaxespad=0)
    plt.subplots_adjust(right=0.75)
    plt.xlabel(r'Частота $\nu$, ГГц')
    plt.ylabel(r'Высота $h$, км')
    # plt.tight_layout()
    plt.tight_layout()
    plt.savefig('112.png', dpi=300)
    plt.show()


def test4():

    plt.figure()
    H, d = 20., 500
    atm = Atmosphere.Standard(H=H, dh=H / d)
    atm.integration_method = 'trapz'
    atm.horizontal_extent = 1.  # km
    atm.approx = True
    _c0 = 0.132574
    _c1 = 2.30215

    print('W\tTeff\tH\ttau_cl\ttau_sum')

    ws = np.asarray([0.15, 0.52, 2., 4.73])
    t_eff = np.asarray([2.9, -2.0, -2, -14.1])
    for w, tcl in zip(ws, t_eff):
        h = np.power(w / _c0, 1. / _c1)
        atm.liquid_water = Cloudiness3D(kilometers=(1, 1, H), nodes=(1, 1, d), clouds_bottom=1.0).liquid_water(
            np.asarray([[h]]), const_w=False, _w=lambda _H: _c0 * np.power(_H, _c1)
        )
        atm.effective_cloud_temperature = tcl
        tau_cl = atm.opacity.liquid_water(36)[0, 0]
        tau_sum = atm.opacity.summary(36)[0, 0]
        print('{}\t{}\t{}\t{}\t{}'.format(w, tcl, h, tau_cl, tau_sum))


def test5():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12})
    matplotlib.rcParams.update({'font.family': 'serif'})
    matplotlib.rcParams.update({'text.latex.preamble':
                                    r'\usepackage[russian]{babel}\usepackage{amsmath}'})

    plt.figure()
    H, d = 20., 500
    sa = Atmosphere.Standard(H=H, dh=H / d)
    sa.integration_method = 'boole'
    sa.horizontal_extent = 1.  # km
    sa.approx = False
    gamma_o, gamma_rho = [], []
    diap = np.arange(5, 350, 1)
    for nu in diap:
        print(nu)
        gamma_o.append(sa.attenuation.oxygen(nu)[0])
        gamma_rho.append(sa.attenuation.water_vapor(nu)[0])
    plt.plot(diap, gamma_o, color='crimson', label='Кислород', ls='--', lw=2)
    plt.plot(diap, gamma_rho, color='darkblue', label='Водяной пар', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(axis='both', ls=':')
    xticks = [10, 22.2, 50, 70, 119, 183.3, 325]
    plt.xticks(ticks=xticks, labels=xticks, rotation=45, fontsize='12')
    plt.yticks(fontsize='12')
    plt.xlabel(r'Частота $\nu$, ГГц')
    plt.ylabel(r'дБ$\cdot$км$^{-1}$')
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig('103-2.png', dpi=300)
    plt.show()


def test90():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 11})
    # matplotlib.rcParams.update({'font.family': 'serif'})
    # matplotlib.rcParams.update({'text.latex.preamble':
    #                                 r'\usepackage[russian]{babel}\usepackage{amsmath}'})

    _H, _d = 30., 1000
    sa25 = Atmosphere.Standard(H=_H, dh=_H / _d, T0=25)
    sa5 = Atmosphere.Standard(H=_H, dh=_H / _d, T0=5)

    # plt.figure()
    # plt.plot(sa25.temperature, np.linspace(0, _H, _d))
    # plt.plot(sa5.temperature, np.linspace(0, _H, _d))
    # plt.show()

    sa25.integration_method = 'trapz'
    sa25.horizontal_extent = 1.  # km
    sa25.approx = False
    sa5.integration_method = 'trapz'
    sa5.horizontal_extent = 1.  # km
    sa5.approx = False

    from scipy.signal import savgol_filter

    colors = ['darkblue', (1, 100 / 255, 0), (60 / 255, 170 / 255, 60 / 255), (0, 157 / 255, 146 / 255), 'black',
              'indigo', 'crimson']
    ls = ['--', '--', '-.', '-', '-', '-', '-.']

    plt.figure()

    for i, nu in enumerate([20, 20.4, 21.2, 21.6, 22.235, 22.8, 23.6]):
        s25 = Staelin(sa25, nu)
        s5 = Staelin(sa5, nu)
        h = np.linspace(0, _H, _d)

        s25 = s25 / np.max(s25)
        s5 = s5 / np.max(s5)

        if nu == 22.235:
            plt.fill_betweenx(h, savgol_filter(s5, 100, 3), savgol_filter(s25, 100, 3), linewidth=2, color=colors[i], alpha=0.8,
                              label='{:.3f} ГГц'.format(nu), zorder=9999)
        else:
            plt.fill_betweenx(h, savgol_filter(s5, 100, 3), savgol_filter(s25, 100, 3), color=colors[i], alpha=0.8,
                              label='{:.1f} ГГц'.format(nu))
    plt.legend(frameon=False)
    plt.xlim((-0.1, 1.6))
    ax = plt.gca()
    xticklabels = ['', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '', '', '', '']
    ax.set_xticklabels(xticklabels)

    plt.xlabel(r'Значения $s$')
    plt.ylabel(r'Высота $h$, км')

    plt.savefig('002.png', dpi=300)
    plt.show()


def test22():
    plt.figure()

    _H, _d = 20., 500
    sa = Atmosphere.Standard(H=_H, dh=_H / _d)
    sa.integration_method = 'trapz'
    sa.horizontal_extent = 1.  # km
    sa.approx = False
    # sa.effective_cloud_temperature = -2.

    frequencies = np.linspace(10, 300., 500)

    theta = None
    # theta = 30. * np.pi / 180
    srf = SmoothWaterSurface(polarization='H')
    # srf.angle = theta

    linestyles = ['-', '-.', '--']
    h = [1, 2, 3]
    __const, __tb = [], []
    for i, H in enumerate(h):

        _c0 = 0.132574
        _c1 = 2.30215
        # H = np.power(W / _c0, 1. / _c1)
        sa.liquid_water = CloudinessColumn(kilometers_z=_H, nodes_z=_d, clouds_bottom=1.1).liquid_water(
            H, const_w=True,
        )
        _const = satellite.brightness_temperatures(frequencies, sa, srf,
                                                   cosmic=True, n_workers=8)
        __const.append(_const[:, 0, 0])

        sa.liquid_water = CloudinessColumn(kilometers_z=_H, nodes_z=_d, clouds_bottom=1.1).liquid_water(
            H, const_w=False,
        )
        # start = time.time()
        tb = satellite.brightness_temperatures(frequencies, sa, srf,
                                               cosmic=True, n_workers=8)
        # stop = time.time()
        # print(stop - start)
        # tb = sa.downward.brightness_temperatures(frequencies, background=False, n_workers=8)
        __tb.append(tb[:, 0, 0])

        plt.plot(frequencies, tb[:, 0, 0],
                 linestyle=linestyles[i], color='black')
        print('\rH = {} km ready'.format(H), end='  ', flush=True)

    import dill
    with open('mazin_vs_const_10-300GHz_H20km_d500_trapz_cl.bottom1.1km.data', 'wb') as dump:
        dill.dump((frequencies, __const, __tb), dump)

    plt.xlabel(r'Frequency $\nu$, GHz')
    plt.ylabel(r'Brightness temperature, К')
    plt.xscale('log')
    xticks = [10, 20, 30, 50, 90, 183, 300]
    plt.xticks(ticks=xticks, labels=xticks)
    plt.legend(frameon=False)
    plt.savefig('mazin_H20km_trapz_effpng', dpi=300)
    # plt.savefig('wconst_H20km.eps')
    plt.show()


if __name__ == '__main__':

    # test1()
    # ex1()
    # ex3()
    # test2()
    # ex4()
    # test3()
    # test4()
    # test5()
    # test21()
    # test90()
    test22()
