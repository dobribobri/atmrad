# -*- coding: utf-8 -*-
from typing import Tuple, Union, List
from cpu.core.types import TensorLike, Tensor1D_or_2D
import cpu.core.math as math
from cpu.atmosphere import Atmosphere, avg
from cpu.weight_funcs import krho, kw
import numpy as np

"""
Обратные задачи
"""


class QW:
    """
    Восстановление интегральных параметров влагосодержания атмосферы
    """

    class downward:
        """
        По нисходящему излучению
        """
        @staticmethod
        def dual_freq(freqs: Union[Tuple[float, float], List[float], np.ndarray],
                      brts: TensorLike,
                      T0: Union[float, Tensor1D_or_2D],
                      P0: Union[float, Tensor1D_or_2D],
                      rho0: Union[float, Tensor1D_or_2D]) -> \
                Tuple[Union[float, Tensor1D_or_2D], Union[float, Tensor1D_or_2D]]:
            """
            Двухчастотный метод восстановления Q и W - полной массы водяного пара и интегрального водозапаса облаков

            :param freqs: две частоты в ГГц
            :param brts: яркостные температуры в зените - (2,) или (2, N) или (2, N, M)
            :param T0: приповерхностная температура воздуха - () или (N,) или (N, M)
            :param P0: давление на уровне поверхности - () или (N,) или (N, M)
            :param rho0: приповерхностное значение абсолютной влажности - () или (N,) или (N, M)
            :return: Q, W
            """
            sa = Atmosphere.Standard(T0, P0, rho0)
            t_avg = math.as_tensor([avg.downward.T(sa, f) for f in freqs])
            t_avg = math.move_axis(t_avg, 0, -1)    # frequencies last

            brts = math.move_axis(brts, 0, -1)      # frequencies last

            tau_e = math.log(t_avg - sa.T_cosmic) - math.log(t_avg - brts)
            # shape of tau_e can be (2,) or (N, 2) or (N, M, 2)

            k_rho = [krho(sa, f) for f in freqs]
            k_w = [kw(sa, f) for f in freqs]
            M = math.transpose(math.as_tensor([k_rho, k_w]))    # always (2, 2)

            if math.rank(tau_e) == 2:   # (N, 2)
                n, _ = math.shape(tau_e)
                M = math.as_tensor([M] * n)     # (N, 2, 2)
            if math.rank(tau_e) == 3:   # (N, M, 2)
                n, m, _ = math.shape(tau_e)
                M = math.as_tensor([[M] * m] * n)   # (N, M, 2, 2)

            tau_o = math.as_tensor([sa.opacity.oxygen(f) for f in freqs])
            tau_o = math.move_axis(tau_o, 0, -1)    # also (2,) or (N, 2) or (N, M, 2)

            Q, W = math.linalg_solve(M, tau_e - tau_o).tolist()
            return Q, W

        @staticmethod
        def multi_freq(freqs: Union[Tuple[float, float], List[float], np.ndarray],
                      brts: TensorLike,
                      T0: Union[float, Tensor1D_or_2D],
                      P0: Union[float, Tensor1D_or_2D],
                      rho0: Union[float, Tensor1D_or_2D]) -> \
                Tuple[Union[float, Tensor1D_or_2D], Union[float, Tensor1D_or_2D]]:
            """
            Многочастотный метод восстановления Q и W - полной массы водяного пара и интегрального водозапаса облаков

            :param freqs: список частот в ГГц
            :param brts: яркостные температуры в зените - (F,) или (F, N) или (F, N, M)
            :param T0: приповерхностная температура воздуха - () или (N,) или (N, M)
            :param P0: давление на уровне поверхности - () или (N,) или (N, M)
            :param rho0: приповерхностное значение абсолютной влажности - () или (N,) или (N, M)
            :return: Q, W
            """
            sa = Atmosphere.Standard(T0, P0, rho0)
            t_avg = math.as_tensor([avg.downward.T(sa, f) for f in freqs])
            t_avg = math.move_axis(t_avg, 0, -1)  # frequencies last

            brts = math.move_axis(brts, 0, -1)  # frequencies last

            tau_e = math.log(t_avg - sa.T_cosmic) - math.log(t_avg - brts)  # frequencies last

            k_rho = np.asarray([krho(sa, f) for f in freqs])
            k_w = np.asarray([kw(sa, f) for f in freqs])
            A = [[np.sum(k_rho * k_rho), np.sum(k_rho * k_w)],
                 [np.sum(k_rho * k_w), np.sum(k_w * k_w)]]
            A = math.as_tensor(A)

            tau_o = math.as_tensor([sa.opacity.oxygen(f) for f in freqs])
            tau_o = math.move_axis(tau_o, 0, -1)  # frequencies last

            b_ = tau_e - tau_o
            b_ = math.move_axis(b_, -1, 0)  # frequencies first
            b = [math.sum_(b_ * krho)]

