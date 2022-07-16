# -*- coding: utf-8 -*-
from collections import Counter
from cpu.atmosphere import Atmosphere
from cpu.cloudiness import Plank3D, Cloudiness3D, CloudinessColumn
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
# from mpl_toolkits.mplot3d import Axes3D
#
#
# def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
#     z = np.linspace(1.5, height_z, 50)
#     theta = np.linspace(0, 2*np.pi, 50)
#     theta_grid, z_grid=np.meshgrid(theta, z)
#     x_grid = radius*np.cos(theta_grid) + center_x
#     y_grid = radius*np.sin(theta_grid) + center_y
#     return x_grid,y_grid,z_grid


if __name__ == '__main__':
    X = 50     # km
    res = 300
    percentage = 0.7
    seed = 42

    # distribution parameters
    distributions = [
        {'name': 'L2Z', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 1.5, 'beta': -0.9, 'cl_bottom': 1.2192},
        {'name': 'L2E', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 1.5, 'beta': 0.3, 'cl_bottom': 1.2192},
        {'name': 'L2B', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 0.93, 'beta': -0.9, 'cl_bottom': 1.2192},
        {'name': 'L2', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 0.93, 'beta': 0.3, 'cl_bottom': 1.2192},
        {'name': 'L3', 'alpha': 1.485, 'Dm': 4.020, 'dm': 0.03048, 'eta': 0.76, 'beta': -0.3, 'cl_bottom': 1.3716},
        {'name': 'T7', 'alpha': 1.35, 'Dm': 3.733, 'dm': 0.04572, 'eta': 1.2, 'beta': 0.0, 'cl_bottom': 1.24968},
        {'name': 'T6', 'alpha': 1.398, 'Dm': 3.376, 'dm': 0.03048, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 1.0668},
        {'name': 'T8', 'alpha': 1.485, 'Dm': 4.02, 'dm': 0.06096, 'eta': 1.2, 'beta': 0.4, 'cl_bottom': 1.3716},
        {'name': 'T9', 'alpha': 2.485, 'Dm': 2.656, 'dm': 0.04572, 'eta': 1.3, 'beta': 0.3, 'cl_bottom': 1.40208},

        {'name': 'L1', 'alpha': 3.853, 'Dm': 1.448, 'dm': 0.01524, 'eta': 0.98, 'beta': 0.0, 'cl_bottom': 0.54864},
        {'name': 'T5', 'alpha': 2.051, 'Dm': 2.574, 'dm': 0.02286, 'eta': 0.85, 'beta': -0.13, 'cl_bottom': 1.11252},
        {'name': 'T3', 'alpha': 2.361, 'Dm': 2.092, 'dm': 0.01524, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 0.82296},
        {'name': 'T4', 'alpha': 2.703, 'Dm': 2.094, 'dm': 0.02286, 'eta': 0.8, 'beta': 0.0, 'cl_bottom': 0.9144},
        {'name': 'T2', 'alpha': 4.412, 'Dm': 1.126, 'dm': 0.01524, 'eta': 0.97, 'beta': 0.0, 'cl_bottom': 0.70104},
        {'name': 'T1', 'alpha': 9.07, 'Dm': 0.80485, 'dm': 0.01524, 'eta': 0.89, 'beta': 0.0, 'cl_bottom': 0.67056},
    ]

    distr = distributions[0]
    alpha, Dm, dm, eta, beta, cl_bottom = \
        distr['alpha'], distr['Dm'], distr['dm'], distr['eta'], distr['beta'], distr['cl_bottom']

    xi = -np.exp(-alpha * Dm) * (((alpha * Dm) ** 2) / 2 + alpha * Dm + 1) + \
        np.exp(-alpha * dm) * (((alpha * dm) ** 2) / 2 + alpha * dm + 1)
    print('xi ', xi)

    K = 2 * np.power(alpha, 3) * (X * X * percentage) / (np.pi * xi)
    print(K)

    p = Plank3D(kilometers=(X, X, 20.), nodes=(res, res, 500), clouds_bottom=cl_bottom)
    clouds = p.generate_clouds(
        Dm=Dm, dm=dm, K=K, alpha=alpha, beta=beta, eta=eta, seed=seed, timeout=1., verbose=True
    )

    n_model = K / alpha * (np.exp(-alpha * dm) - np.exp(-alpha * Dm))
    print('N ', len(clouds), n_model)

    #############################################################
    atmosphere = Atmosphere.Standard()
    hmap = p.height_map2d_(clouds)
    atmosphere.liquid_water = p.liquid_water_(hmap2d=hmap, const_w=False)

    print('% ', np.count_nonzero(hmap) / (res * res) * 100.,
          np.sum(np.pi * np.power(np.array([cloud.rx for cloud in clouds]), 2)) / (X * X) * 100.)

    print('Wmax\t', np.max(atmosphere.W))

    # # ################# 3D #################
    # #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # for cloud in clouds[:10]:
    #     Xc, Yc, Zc = data_for_cylinder_along_z(cloud.x, cloud.y, cloud.rx, cloud.height)
    #     ax.plot_surface(Xc, Yc, Zc, color='gray')
    # ax.set_zlim((0, 10))
    # plt.show()
    # #
    # # ######################################

    plt.figure()
    plt.imshow(atmosphere.W)
    h, w = atmosphere.W.shape
    plt.title(r'Liquid Water Content, kg/m$^2$ {:.2f}%'.format(np.count_nonzero(hmap) / (h * w) * 100.))
    plt.xlabel('km')
    # ticks_pos = np.asarray([30, 60, 90, 120, 150, 180, 210, 240, 270])
    ticks_pos = np.linspace(0, res, 10)
    ticks_labels = np.round(ticks_pos / res * X, decimals=0)
    ticks_labels = [int(i) for i in ticks_labels]
    plt.xticks(ticks_pos, ticks_labels)
    plt.ylabel('km')
    plt.yticks(ticks_pos, ticks_labels)
    plt.colorbar()
    plt.savefig('pic.K{}_Dm{:.1f}_alpha{:.2f}_beta{:.2f}_eta{:.1f}.png'.format(K, Dm, alpha, beta, eta), dpi=300)
    plt.show()
    #############################################################

    with open('Dm{}_K{}_alpha{}_seed{}.clouds.txt'.format(
        np.round(Dm, decimals=0), np.round(K, decimals=0), np.round(alpha, decimals=1), seed
    ), 'w') as file:
        file.write('R\tX\tY\n')
        for r, x, y in zip([c.rx for c in clouds], [c.x for c in clouds], [c.y for c in clouds]):
            line = '{:.16f}\t{:.16f}\t{:.16f}\n'.format(r, x, y)
            file.write(line)

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
    ax.set_ylabel('Number of clouds')

    ax.scatter(H, N, color='crimson', marker='+', label='Number of clouds $n(H)$')

    coeff, _ = curve_fit(lambda t, a, b, c: a + b * np.sqrt(t) + c * t, H, N)
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
    ax2.set_ylabel('Area covered by clouds, km$^2$')

    ax2.scatter(H, S, color='darkblue', marker='.', label='Area covered by clouds')

    coeff, _ = curve_fit(lambda t, a, b, c: a + b * np.exp(-t) + c * np.exp(-2 * t), H, S)
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
    ax2.plot(H, [np.sum(coeff * np.array([1, np.exp(-h), np.exp(-2 * h)])) for h in H], color='darkblue',
             linestyle='--',
             label=label)
    ax2.legend(loc='upper left', frameon=False)

    full_square = np.sum(S)
    print(full_square)
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

    import dill
    with open('stat_L2Z_procentage70.data', 'wb') as dump:
        dill.dump((H, N, S), dump)
