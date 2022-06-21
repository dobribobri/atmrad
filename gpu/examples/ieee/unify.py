
# 'W': {
#     # 'map': W,
#     'total_max': np.max(W),
#     'WINI': WINI,
#
#     'WBRT': WBRT,
#     'WSOL': WSOL,
#     'DWSB': DWSB,
#     'DWBI': DWBI,
#     'DWSI': DWSI,
#
#     'DWSBI': DWSBI,
#     'DWBII': DWBII,
#     'DWSII': DWSII,
# },
#
# 'brightness_temperature': {
#     # 'maps': brts,
#     'BRTC': BRTC,
#     'SOLD': SOLD,
#     'DTSB': DTSB,
# }

import glob
import dill


keys = ['name', 'seed', 'required_percentage', 'K', 'alpha', 'Dm', 'd_min', 'eta', 'beta', 'cl_bottom', 'xi',
        'cover_percentage', 'cover_percentage_d', 'sky_cover', 'sky_cover_d', 'n_analytical', 'n_fact']

stats = ['mean', 'min', 'max', 'var', 'std', 'range']

with open('db.txt', 'w') as db:

    s = ' '.join(keys + ['w_total_max', 'kernel_nodes', 'kernel_km']
                      + ['WINI_{}'.format(t) for t in stats]
                      + ['freq']

                      + ['BRTC_{}'.format(t) for t in stats] + ['SOLD_{}'.format(t) for t in stats]
                      + ['DTSB_{}'.format(t) for t in stats]

                      + ['WBRT_{}'.format(t) for t in stats] + ['WSOL_{}'.format(t) for t in stats]
                      + ['DWSB_{}'.format(t) for t in stats]
                      + ['DWBI_{}'.format(t) for t in stats] + ['DWSI_{}'.format(t) for t in stats]
                      + ['DWSBI_{}'.format(t) for t in stats]
                      + ['DWBII_{}'.format(t) for t in stats] + ['DWSII_{}'.format(t) for t in stats]
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

            s_init += '{:.4f} '.format(data['W']['total_max'])

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
                        ##########################################################################

                        s += '{:.4f} '.format(data['brightness_temperature']['BRTC'][key][nu][k])

                        s += '{:.4f} '.format(data['brightness_temperature']['SOLD'][key][nu][k])

                        s += '{:.4f} '.format(data['brightness_temperature']['DTSB'][key][nu][k])

                        ##########################################################################

                        s += '{:.4f} '.format(data['W']['WBRT'][key][nu][k])

                        s += '{:.4f} '.format(data['W']['WSOL'][key][nu][k])

                        s += '{:.4f} '.format(data['W']['DWSB'][key][nu][k])

                        s += '{:.4f} '.format(data['W']['DWBI'][key][nu][k])
                        s += '{:.4f} '.format(data['W']['DWSI'][key][nu][k])
                        s += '{:.4f} '.format(data['W']['DWSBI'][key][nu][k])
                        s += '{:.4f} '.format(data['W']['DWBII'][key][nu][k])
                        s += '{:.4f} '.format(data['W']['DWSII'][key][nu][k])

                    db.write(s + '\n')
