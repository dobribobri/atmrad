
import glob
import dill


keys = ['name', 'seed', 'required_percentage', 'K', 'alpha', 'Dm', 'd_min', 'eta', 'beta', 'cl_bottom', 'xi',
        'cover_percentage', 'cover_percentage_d', 'sky_cover', 'sky_cover_d', 'n_analytical', 'n_fact']

with open('db.txt', 'w') as db:

    s = ' '.join(keys + ['w_total_max',
                         'kernel',
                         'w_mean', 'w_min', 'w_max', 'w_var', 'w_std', 'w_range',
                         'freq',
                         'delta_mean', 'delta_min', 'delta_max', 'delta_var', 'delta_std', 'delta_range'])
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

            s_init += '{} '.format(data['W']['total_max'])

            conv_w_stats = data['W']['conv_stats']
            delta_conv_stats = data['brightness_temperature']['delta_stats']

            frequencies = data['frequencies']
            kernels = data['kernels']
            for k, kernel in enumerate(kernels):
                s_pre = s_init
                s_pre += '{} '.format(kernel)
                for key in conv_w_stats.keys():
                    s_pre += '{} '.format(conv_w_stats[key][k])

                for nu in frequencies:
                    s = s_pre
                    s += '{} '.format(nu)
                    for key in delta_conv_stats:
                        s += '{} '.format(delta_conv_stats[key][nu][k])

                    db.write(s + '\n')
