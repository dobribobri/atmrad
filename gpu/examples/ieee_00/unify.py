
import glob
import dill


keys = ['name', 'seed', 'required_percentage', 'K', 'alpha', 'Dm', 'd_min', 'eta', 'beta', 'cl_bottom', 'xi',
        'cover_percentage', 'cover_percentage_d', 'sky_cover', 'sky_cover_d', 'n_analytical', 'n_fact']

stats = ['mean', 'min', 'max', 'var', 'std', 'range']

with open('db.txt', 'w') as db:

    s = ' '.join(keys + ['w_total_{}'.format(t) for t in stats] + ['q_total_mean']
                      + ['kernel_nodes', 'kernel_km']
                      + ['WINI_{}'.format(t) for t in stats]
                      + ['QINI']
                      + ['freq']

                      + ['BRTC_{}'.format(t) for t in stats] + ['SOLD_{}'.format(t) for t in stats]
                      + ['DTSB_{}'.format(t) for t in stats]

                      + ['WBRT_{}'.format(t) for t in stats] + ['WSOL_{}'.format(t) for t in stats]
                      + ['DWSB_{}'.format(t) for t in stats]
                      + ['DWBI_{}'.format(t) for t in stats] + ['DWSI_{}'.format(t) for t in stats]
                      + ['DWSBI_{}'.format(t) for t in stats]
                      + ['DWBII_{}'.format(t) for t in stats] + ['DWSII_{}'.format(t) for t in stats]

                      + ['QBRT_{}'.format(t) for t in stats] + ['QSOL_{}'.format(t) for t in stats]
                      + ['DQSB_{}'.format(t) for t in stats]
                      + ['DQBI_{}'.format(t) for t in stats] + ['DQSI_{}'.format(t) for t in stats]
                      + ['DQSBI_{}'.format(t) for t in stats]
                      + ['DQBII_{}'.format(t) for t in stats] + ['DQSII_{}'.format(t) for t in stats]

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
            s_init += '{} '.format(data['Q']['total_mean'])

            frequencies = data['frequencies']
            kernels = data['kernels']
            for k, kernel in enumerate(kernels):
                s_pre = s_init
                s_pre += '{} '.format(kernel)
                s_pre += '{} '.format(int(kernel) // 6)

                for key in stats:
                    s_pre += '{} '.format(data['W']['WINI'][key][k])
                s_pre += '{} '.format(data['Q']['QINI'][k])

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
                        s += '{} '.format(data['Q']['QBRT'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['Q']['QSOL'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['Q']['DQSB'][key][nu][k])

                    for key in stats:
                        s += '{} '.format(data['Q']['DQBI'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['Q']['DQSI'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['Q']['DQSBI'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['Q']['DQBII'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['Q']['DQSII'][key][nu][k])

                        ##########################################################################

                    for key in stats:
                        s += '{} '.format(data['opacity']['OPBC'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['opacity']['OPSD'][key][nu][k])
                    for key in stats:
                        s += '{} '.format(data['opacity']['DOSB'][key][nu][k])

                    db.write(s + '\n')
