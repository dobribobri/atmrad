# data = {
#     'name': distr['name'],
#     'part': ID,
#     'filename': '{}_{}.part'.format(distr['name'], str(ID).zfill(4)),
#
#     'H': H,
#     'd': d,
#     'X': X,
#     'res': res,
#
#     'angle': angle,
#     'incline': incline,
#     'integration_method': integration_method,
#     'T0': T0,
#     'P0': P0,
#     'rho0': rho0,
#     'surface_temperature': surface_temperature,
#     'surface_salinity': surface_salinity,
#
#     'required_percentage': np.round(required_percentage * 100., decimals=3),
#     'cover_percentage': np.round(cover_percentage * 100., decimals=3),
#     'cover_percentage_d': np.round(cover_percentage_d * 100., decimals=3),
#     'sky_cover': sky_cover,
#     'sky_cover_d': sky_cover_d,
#
#     'seed': seed,
#
#     'K': np.round(K, decimals=3),
#     'alpha': alpha,
#     'Dm': Dm,
#     'd_min': dm,
#     'xi': xi,
#
#     'n_analytical': N_analytical,
#     'n_fact': N_fact,
#
#     'eta': eta,
#     'beta': beta,
#     'cl_bottom': cl_bottom,
#
#     # 'clouds': clouds,
#     # 'heights': {
#     #     'map': hmap,
#     # },
#
#     'mu0': mu0,
#     'psi0': psi0,
#     'c0': _c0,
#     'c1': _c1,
#
#     # 'liquid_water': atmosphere.liquid_water,
#
#     'kernels': np.asarray(_kernels),
#
#     'W': {
#         # 'map': W,
#         'total_max': np.max(W),
#         'total_mean': np.mean(W),
#         'total_min': np.min(W),
#         'total_var': np.var(W),
#         'total_std': np.std(W),
#         'total_range': np.max(W) - np.min(W),
#
#         'WINI': WINI,
#
#         'WBRT': WBRT,
#         'WSOL': WSOL,
#         'DWSB': DWSB,
#         'DWBI': DWBI,
#         'DWSI': DWSI,
#
#         'DWSBI': DWSBI,
#         'DWBII': DWBII,
#         'DWSII': DWSII,
#     },
#
#     'frequencies': frequencies,
#
#     'brightness_temperature': {
#         # 'maps': brts,
#         'BRTC': BRTC,
#         'SOLD': SOLD,
#         'DTSB': DTSB,
#     },
#
#     'opacity': {
#         'OPBC': OPBC,
#         'OPSD': OPSD,
#         'DOSB': DOSB,
#     }
# }

import glob
import dill


keys = ['name', 'seed', 'required_percentage', 'K', 'alpha', 'Dm', 'd_min', 'eta', 'beta', 'cl_bottom', 'xi',
        'cover_percentage', 'cover_percentage_d', 'sky_cover', 'sky_cover_d', 'n_analytical', 'n_fact']

stats = ['mean', 'min', 'max', 'var', 'std', 'range']

with open('db.txt', 'w') as db:

    s = ' '.join(keys + ['w_total_{}'.format(t) for t in stats]
                      + ['kernel_nodes', 'kernel_km']
                      + ['WINI_{}'.format(t) for t in stats]
                      + ['freq']

                      + ['BRTC_{}'.format(t) for t in stats] + ['SOLD_{}'.format(t) for t in stats]
                      + ['DTSB_{}'.format(t) for t in stats]

                      + ['WBRT_{}'.format(t) for t in stats] + ['WSOL_{}'.format(t) for t in stats]
                      + ['DWSB_{}'.format(t) for t in stats]
                      + ['DWBI_{}'.format(t) for t in stats] + ['DWSI_{}'.format(t) for t in stats]
                      + ['DWSBI_{}'.format(t) for t in stats]
                      + ['DWBII_{}'.format(t) for t in stats] + ['DWSII_{}'.format(t) for t in stats]

                      + ['OPBC_{}'.format(t) for t in stats] + ['OPSD_{}'.format(t) for t in stats]
                      + ['DOSB_{}'.format(t) for t in stats]
                 )
    db.write(s + '\n')

    parts = glob.glob('*.part')
    N = len(parts)
    for i, fname in enumerate(parts):
        print('\r{:.2f}%'.format((i + 1) / N * 100.), end='  ', flush=True)
        with open(fname, 'rb') as file:
            data = dill.load(file)

            s_init = ''
            for key in keys:
                s_init += '{} '.format(data[key])

            for key in stats:
                s_init += '{} '.format(data['W']['total_{}'.format(key)])

            frequencies = data['frequencies']
            kernels = data['kernels']
            for k, kernel in enumerate(kernels):
                s_pre = s_init
                s_pre += '{} '.format(kernel)
                s_pre += '{} '.format(int(kernel) // 6)

                for key in stats:
                    s_pre += '{} '.format(data['W']['WINI'][key][k])

                for nu in frequencies:
                    s = s_pre
                    s += '{} '.format(nu)

                    for key in stats:
                        s += '{} '.format(data['brightness_temperature']['BRTC'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['brightness_temperature']['SOLD'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['brightness_temperature']['DTSB'][key][nu][k])

                        ##########################################################################

                    for key in stats:
                        s += '{} '.format(data['W']['WBRT'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['W']['WSOL'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['W']['DWSB'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['W']['DWBI'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['W']['DWSI'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['W']['DWSBI'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['W']['DWBII'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['W']['DWSII'][key][nu][k])

                        ##########################################################################

                    for key in stats:
                        s += '{} '.format(data['opacity']['OPBC'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['opacity']['OPSD'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['opacity']['DOSB'][key][nu][k])

                    db.write(s + '\n')
