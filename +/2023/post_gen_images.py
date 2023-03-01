
import os
import dill
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class Data:
    def __init__(self, column_names, data):

        self.names = column_names
        self.data = data

        n = {}
        for key, value in enumerate(self.names):
            n[value] = key
        self.n = n

    def get(self, *args) -> np.ndarray:
        out = []
        for arg in args:
            out.append(self.data[:, self.n[arg]])
        out = np.asarray(out)
        if len(args) == 1:
            return out[0]
        return out

    def dist(self, prefix: str) -> 'Data':
        cond = np.asarray([str(distr_name).find(prefix) + 1
                           for distr_name in self.get('distr_name')], dtype=bool)
        return Data(column_names=self.names, data=self.data[cond])

    def select(self, **kwargs) -> 'Data':
        cond = np.asarray([True for _ in range(len(self.data))])
        for key, value in kwargs.items():
            d = self.get(str(key)).astype(type(value))
            cond = np.isclose(d, value) & cond
        return Data(column_names=self.names, data=self.data[cond])


def means(arr) -> np.ndarray:
    return np.asarray([val.mean for val in arr])


def mins(arr) -> np.ndarray:
    return np.asarray([val.min for val in arr])


def maxs(arr) -> np.ndarray:
    return np.asarray([val.max for val in arr])


def bind(*arrays):
    return tuple(np.asarray(sorted(list(zip(*arrays)), key=lambda e: e[0])).T)


linestyles = {
    'loosely dotted': (0, (1, 10)),
    'dotted': (0, (1, 1)),
    'densely dotted': (0, (1, 1)),

    'loosely dashed': (0, (5, 10)),
    'dashed': (0, (5, 5)),
    'densely dashed': (0, (5, 1)),

    'loosely dashdotted': (0, (3, 10, 1, 10)),
    'dashdotted': (0, (3, 5, 1, 5)),
    'densely dashdotted': (0, (3, 1, 1, 1)),

    'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
    'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
    'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
}


T1 = 0.2
T2 = 0.2
T3 = 0.2789473684210526
T4 = 0.30526315789473685
T5 = 0.35789473684210527
T6 = 0.48947368421052634
T7 = 0.30526315789473685
T8 = 0.2
T9 = 0.2
L1 = 0.4368421052631579
L2 = 0.6473684210526316
L3 = 0.30526315789473685

base_distributions = [

    {'name': 'T8', 'alpha': 1.485, 'Dm': 4.02, 'dm': 0.06096, 'eta': 1.2, 'beta': 0.4, 'cl_bottom': 1.3716,
     'st': T8},
    {'name': 'T7', 'alpha': 1.35, 'Dm': 3.733, 'dm': 0.04572, 'eta': 1.2, 'beta': 0.0, 'cl_bottom': 1.24968,
     'st': T7},
    {'name': 'L2', 'alpha': 1.411, 'Dm': 4.026, 'dm': 0.02286, 'eta': 0.93, 'beta': 0.3, 'cl_bottom': 1.2192,
     'st': L2},

    {'name': 'L3', 'alpha': 1.485, 'Dm': 4.020, 'dm': 0.03048, 'eta': 0.76, 'beta': -0.3, 'cl_bottom': 1.3716,
     'st': L3},
    {'name': 'T6', 'alpha': 1.398, 'Dm': 3.376, 'dm': 0.03048, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 1.0668,
     'st': T6},

    {'name': 'T9', 'alpha': 2.485, 'Dm': 2.656, 'dm': 0.04572, 'eta': 1.3, 'beta': 0.3, 'cl_bottom': 1.40208,
     'st': T9},
    {'name': 'T5', 'alpha': 2.051, 'Dm': 2.574, 'dm': 0.02286, 'eta': 0.85, 'beta': -0.13, 'cl_bottom': 1.11252,
     'st': T5},
    {'name': 'T3', 'alpha': 2.361, 'Dm': 2.092, 'dm': 0.01524, 'eta': 0.93, 'beta': -0.1, 'cl_bottom': 0.82296,
     'st': T3},
    {'name': 'T4', 'alpha': 2.703, 'Dm': 2.094, 'dm': 0.02286, 'eta': 0.8, 'beta': 0.0, 'cl_bottom': 0.9144,
     'st': T4},
    {'name': 'L1', 'alpha': 3.853, 'Dm': 1.448, 'dm': 0.01524, 'eta': 0.98, 'beta': 0.0, 'cl_bottom': 0.54864,
     'st': L1},
    {'name': 'T2', 'alpha': 4.412, 'Dm': 1.126, 'dm': 0.01524, 'eta': 0.97, 'beta': 0.0, 'cl_bottom': 0.70104,
     'st': T2},
    {'name': 'T1', 'alpha': 9.07, 'Dm': 0.80485, 'dm': 0.01524, 'eta': 0.89, 'beta': 0.0, 'cl_bottom': 0.67056,
     'st': T1},
]


if __name__ == '__main__':
    with open('post_data_theta0_kernel60_all.bin', 'rb') as dump:
        data = dill.load(dump)

    data = Data(column_names=data[0], data=data[1:])

    print(data.names)

    stat = []

    for distr in base_distributions:

        ###############################################################################################
        # # eta variance, beta fixed
        path = os.path.join('fig', distr['name'], 'eta_variance')
        # path = os.path.join('fig', 'fixed_axes', distr['name'], 'eta_variance')

        # LWC
        savepath = os.path.join(path, 'LWC')
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        for i, p in enumerate(np.unique(data.dist(distr['name']).select(beta=distr['beta']).get('required_percentage'))[:-1]):

            plt.figure()

            plt.title(distr['name'])

            eta, *W = \
                data.dist(distr['name']).select(beta=distr['beta'], required_percentage=p, nu1=22.2, nu2=27).get(
                    'eta', 'Delta_Wrs'
                )
            eta, Delta_Wrs = bind(eta, *W)
            plt.plot(eta, means(Delta_Wrs), marker='x', color='darkblue', linewidth=2, linestyle='-', zorder=100)
            err = np.asarray(list(zip(means(Delta_Wrs) - mins(Delta_Wrs), maxs(Delta_Wrs) - means(Delta_Wrs)))).T
            plt.errorbar(eta, means(Delta_Wrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='darkblue', zorder=99)
            plt.plot(eta[eta == distr['eta']], means(Delta_Wrs)[eta == distr['eta']], marker='o', color='darkblue', zorder=999)
            # if np.isclose(p, distr['st']):
            #     stat.append((distr['name'], '22.2 and 27.2 ГГц', means(Delta_Wrs)[eta == distr['eta']]))

            eta, *W = \
                data.dist(distr['name']).select(beta=distr['beta'], required_percentage=p, nu1=22.2, nu2=36).get(
                    'eta', 'Delta_Wrs'
                )
            eta, Delta_Wrs = bind(eta, *W)
            plt.plot(eta, means(Delta_Wrs), marker='x', color='forestgreen', linewidth=2, linestyle='--', zorder=10)
            err = np.asarray(list(zip(means(Delta_Wrs) - mins(Delta_Wrs), maxs(Delta_Wrs) - means(Delta_Wrs)))).T
            plt.errorbar(eta, means(Delta_Wrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='forestgreen', zorder=9)
            plt.plot(eta[eta == distr['eta']], means(Delta_Wrs)[eta == distr['eta']], marker='o', color='forestgreen', zorder=998)

            eta, *W = \
                data.dist(distr['name']).select(beta=distr['beta'], required_percentage=p, nu1=22.2, nu2=89).get(
                    'eta', 'W_TRUE', 'Delta_Wrs'
                )
            eta, W_TRUE, Delta_Wrs = bind(eta, *W)
            plt.plot(eta, means(Delta_Wrs), marker='x', color='crimson', linewidth=2, linestyle='-.', zorder=1)
            err = np.asarray(list(zip(means(Delta_Wrs) - mins(Delta_Wrs), maxs(Delta_Wrs) - means(Delta_Wrs)))).T
            plt.errorbar(eta, means(Delta_Wrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='crimson', zorder=0)
            plt.plot(eta[eta == distr['eta']], means(Delta_Wrs)[eta == distr['eta']], marker='o', color='crimson', zorder=997)

            plt.xlabel(r'Parameter $\eta$, dimensionless')
            plt.ylabel(r'LWC delta in a resolution element, kg/m$^2$')

            # plt.ylim((-0.09, 1.3))
            plt.text(np.min(eta), 1.1, 'Cloud amount: {:.0f}%'.format(np.round(p * 100, decimals=0)))

            plt.grid(linestyle=':', alpha=0.5)

            f = interp1d(eta, W_TRUE)
            xticks = plt.xticks()[0]
            ax = plt.gca()
            xlim = ax.get_xlim()
            ax2 = ax.twiny()
            ax2.set_xlim(xlim)
            ax2.set_xticklabels(['', *map(lambda val: '{:.2f}'.format(np.round(val, decimals=2)), f(xticks[1:-1])), ''])
            ax2.set_xlabel(r'True overall grid mean LWC, kg/m$^2$')

            plt.savefig(os.path.join(savepath, '{}.png'.format(str(i).zfill(2))), dpi=300)

        command = 'gifski --fps 3 -o ' + os.path.join(path, 'LWC.gif') + ' ' + os.path.join(savepath, '*.png')
        print(command)
        os.system(command)

        # TWV
        savepath = os.path.join(path, 'TWV')
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        for i, p in enumerate(np.unique(data.dist(distr['name']).select(beta=distr['beta']).get('required_percentage'))[:-1]):
            plt.figure()

            plt.title(distr['name'])

            eta, *Q = \
                data.dist(distr['name']).select(beta=distr['beta'], required_percentage=p, nu1=22.2, nu2=27).get(
                    'eta', 'Delta_Qrs'
                )
            eta, Delta_Qrs = bind(eta, *Q)
            plt.plot(eta, means(Delta_Qrs), marker='x', color='darkblue', linewidth=2, linestyle='-', zorder=100)
            err = np.asarray(list(zip(means(Delta_Qrs) - mins(Delta_Qrs), maxs(Delta_Qrs) - means(Delta_Qrs)))).T
            plt.errorbar(eta, means(Delta_Qrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='darkblue', zorder=99)
            plt.plot(eta[eta == distr['eta']], means(Delta_Qrs)[eta == distr['eta']], marker='o', color='darkblue', zorder=999)

            eta, *Q = \
                data.dist(distr['name']).select(beta=distr['beta'], required_percentage=p, nu1=22.2, nu2=36).get(
                    'eta', 'Delta_Qrs'
                )
            eta, Delta_Qrs = bind(eta, *Q)
            plt.plot(eta, means(Delta_Qrs), marker='x', color='forestgreen', linewidth=2, linestyle='--', zorder=10)
            err = np.asarray(list(zip(means(Delta_Qrs) - mins(Delta_Qrs), maxs(Delta_Qrs) - means(Delta_Qrs)))).T
            plt.errorbar(eta, means(Delta_Qrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='forestgreen', zorder=9)
            plt.plot(eta[eta == distr['eta']], means(Delta_Qrs)[eta == distr['eta']], marker='o', color='forestgreen', zorder=998)

            eta, W_TRUE, *Q = \
                data.dist(distr['name']).select(beta=distr['beta'], required_percentage=p, nu1=22.2, nu2=89).get(
                    'eta', 'W_TRUE', 'Delta_Qrs'
                )
            eta, W_TRUE, Delta_Qrs = bind(eta, W_TRUE, *Q)
            plt.plot(eta, means(Delta_Qrs), marker='x', color='crimson', linewidth=2, linestyle='-.', zorder=1)
            err = np.asarray(list(zip(means(Delta_Qrs) - mins(Delta_Qrs), maxs(Delta_Qrs) - means(Delta_Qrs)))).T
            plt.errorbar(eta, means(Delta_Qrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='crimson', zorder=0)
            plt.plot(eta[eta == distr['eta']], means(Delta_Qrs)[eta == distr['eta']], marker='o', color='crimson', zorder=997)

            plt.xlabel(r'Parameter $\eta$, dimensionless')
            plt.ylabel(r'TWV delta in a resolution element, g/cm$^2$')

            # plt.ylim((0.09, -1.3))
            plt.text(np.min(eta), -1.1, 'Cloud amount: {:.0f}%'.format(np.round(p * 100, decimals=0)))

            plt.grid(linestyle=':', alpha=0.5)

            f = interp1d(eta, W_TRUE)
            xticks = plt.xticks()[0]
            ax = plt.gca()
            xlim = ax.get_xlim()
            ax2 = ax.twiny()
            ax2.set_xlim(xlim)
            ax2.set_xticklabels(['', *map(lambda val: '{:.2f}'.format(np.round(val, decimals=2)), f(xticks[1:-1])), ''])
            ax2.set_xlabel(r'True overall grid mean LWC, kg/m$^2$')

            plt.savefig(os.path.join(savepath, '{}.png'.format(str(i).zfill(2))), dpi=300)

        command = 'gifski --fps 3 -o ' + os.path.join(path, 'TWV.gif') + ' ' + os.path.join(savepath, '*.png')
        print(command)
        os.system(command)

        ###############################################################################################
        # # beta variance, eta fixed
        path = os.path.join('fig', distr['name'], 'eta_variance')
        # path = os.path.join('fig', 'fixed_axes', distr['name'], 'beta_variance')

        # LWC
        savepath = os.path.join(path, 'LWC')
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        for i, p in enumerate(np.unique(data.dist(distr['name']).select(eta=distr['eta']).get('required_percentage'))[:-1]):
            plt.figure()

            plt.title(distr['name'])

            beta, *W = \
                data.dist(distr['name']).select(eta=distr['eta'], required_percentage=p, nu1=22.2, nu2=27).get(
                    'beta', 'Delta_Wrs'
                )
            beta, Delta_Wrs = bind(beta, *W)
            plt.plot(beta, means(Delta_Wrs), marker='x', color='darkblue', linewidth=2, linestyle='-', zorder=100)
            err = np.asarray(list(zip(means(Delta_Wrs) - mins(Delta_Wrs), maxs(Delta_Wrs) - means(Delta_Wrs)))).T
            plt.errorbar(beta, means(Delta_Wrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='darkblue', zorder=99)
            plt.plot(beta[beta == distr['beta']], means(Delta_Wrs)[beta == distr['beta']], marker='o', color='darkblue', zorder=999)

            beta, *W = \
                data.dist(distr['name']).select(eta=distr['eta'], required_percentage=p, nu1=22.2, nu2=36).get(
                    'beta', 'Delta_Wrs'
                )
            beta, Delta_Wrs = bind(beta, *W)
            plt.plot(beta, means(Delta_Wrs), marker='x', color='forestgreen', linewidth=2, linestyle='--', zorder=10)
            err = np.asarray(list(zip(means(Delta_Wrs) - mins(Delta_Wrs), maxs(Delta_Wrs) - means(Delta_Wrs)))).T
            plt.errorbar(beta, means(Delta_Wrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='forestgreen', zorder=9)
            plt.plot(beta[beta == distr['beta']], means(Delta_Wrs)[beta == distr['beta']], marker='o', color='forestgreen', zorder=998)

            beta, *W = \
                data.dist(distr['name']).select(eta=distr['eta'], required_percentage=p, nu1=22.2, nu2=89).get(
                    'beta', 'W_TRUE', 'Delta_Wrs'
                )
            beta, W_TRUE, Delta_Wrs = bind(beta, *W)
            plt.plot(beta, means(Delta_Wrs), marker='x', color='crimson', linewidth=2, linestyle='-.', zorder=1)
            err = np.asarray(list(zip(means(Delta_Wrs) - mins(Delta_Wrs), maxs(Delta_Wrs) - means(Delta_Wrs)))).T
            plt.errorbar(beta, means(Delta_Wrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='crimson', zorder=0)
            plt.plot(beta[beta == distr['beta']], means(Delta_Wrs)[beta == distr['beta']], marker='o', color='crimson', zorder=997)

            plt.xlabel(r'Parameter $\beta$, dimensionless')
            plt.ylabel(r'LWC delta in a resolution element, kg/m$^2$')

            # plt.ylim((-0.09, 1.3))
            plt.text(0.25, 1.1, 'Cloud amount: {:.0f}%'.format(np.round(p * 100, decimals=0)))

            plt.grid(linestyle=':', alpha=0.5)

            f = interp1d(beta, W_TRUE)
            xticks = plt.xticks()[0]
            ax = plt.gca()
            xlim = ax.get_xlim()
            ax2 = ax.twiny()
            ax2.set_xlim(xlim)
            ax2.set_xticklabels(['', *map(lambda val: '{:.2f}'.format(np.round(val, decimals=2)), f(xticks[1:-1])), ''])
            ax2.set_xlabel(r'True overall grid mean LWC, kg/m$^2$')

            plt.savefig(os.path.join(savepath, '{}.png'.format(str(i).zfill(2))), dpi=300)

        command = 'gifski --fps 3 -o ' + os.path.join(path, 'LWC.gif') + ' ' + os.path.join(savepath, '*.png')
        print(command)
        os.system(command)

        # TWV
        savepath = os.path.join(path, 'TWV')
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        for i, p in enumerate(np.unique(data.dist(distr['name']).select(eta=distr['eta']).get('required_percentage'))[:-1]):
            plt.figure()

            plt.title(distr['name'])

            beta, *Q = \
                data.dist(distr['name']).select(eta=distr['eta'], required_percentage=p, nu1=22.2, nu2=27).get(
                    'beta', 'Delta_Qrs'
                )
            beta, Delta_Qrs = bind(beta, *Q)
            plt.plot(beta, means(Delta_Qrs), marker='x', color='darkblue', linewidth=2, linestyle='-', zorder=100)
            err = np.asarray(list(zip(means(Delta_Qrs) - mins(Delta_Qrs), maxs(Delta_Qrs) - means(Delta_Qrs)))).T
            plt.errorbar(beta, means(Delta_Qrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='darkblue', zorder=99)
            plt.plot(beta[beta == distr['beta']], means(Delta_Qrs)[beta == distr['beta']], marker='o', color='darkblue', zorder=999)

            beta, *Q = \
                data.dist(distr['name']).select(eta=distr['eta'], required_percentage=p, nu1=22.2, nu2=36).get(
                    'beta', 'Delta_Qrs'
                )
            beta, Delta_Qrs = bind(beta, *Q)
            plt.plot(beta, means(Delta_Qrs), marker='x', color='forestgreen', linewidth=2, linestyle='--', zorder=10)
            err = np.asarray(list(zip(means(Delta_Qrs) - mins(Delta_Qrs), maxs(Delta_Qrs) - means(Delta_Qrs)))).T
            plt.errorbar(beta, means(Delta_Qrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='forestgreen', zorder=9)
            plt.plot(beta[beta == distr['beta']], means(Delta_Qrs)[beta == distr['beta']], marker='o', color='forestgreen', zorder=998)

            beta, W_TRUE, *Q = \
                data.dist(distr['name']).select(eta=distr['eta'], required_percentage=p, nu1=22.2, nu2=89).get(
                    'beta', 'W_TRUE', 'Delta_Qrs'
                )
            beta, W_TRUE, Delta_Qrs = bind(beta, W_TRUE, *Q)
            plt.plot(beta, means(Delta_Qrs), marker='x', color='crimson', linewidth=2, linestyle='-.', zorder=1)
            err = np.asarray(list(zip(means(Delta_Qrs) - mins(Delta_Qrs), maxs(Delta_Qrs) - means(Delta_Qrs)))).T
            plt.errorbar(beta, means(Delta_Qrs), err,
                         fmt='x', markersize=6, capsize=5, elinewidth=0, alpha=1, color='crimson', zorder=0)
            plt.plot(beta[beta == distr['beta']], means(Delta_Qrs)[beta == distr['beta']], marker='o', color='crimson', zorder=997)

            plt.xlabel(r'Parameter $\beta$, dimensionless')
            plt.ylabel(r'TWV delta in a resolution element, g/cm$^2$')

            # plt.ylim((0.09, -1.3))
            plt.text(0.25, -1.1, 'Cloud amount: {:.0f}%'.format(np.round(p * 100, decimals=0)))

            plt.grid(linestyle=':', alpha=0.5)

            f = interp1d(beta, W_TRUE)
            xticks = plt.xticks()[0]
            ax = plt.gca()
            xlim = ax.get_xlim()
            ax2 = ax.twiny()
            ax2.set_xlim(xlim)
            ax2.set_xticklabels(['', *map(lambda val: '{:.2f}'.format(np.round(val, decimals=2)), f(xticks[1:-1])), ''])
            ax2.set_xlabel(r'True overall grid mean LWC, kg/m$^2$')

            plt.savefig(os.path.join(savepath, '{}.png'.format(str(i).zfill(2))), dpi=300)

        command = 'gifski --fps 3 -o ' + os.path.join(path, 'TWV.gif') + ' ' + os.path.join(savepath, '*.png')
        print(command)
        os.system(command)