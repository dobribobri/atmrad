#  -*- coding: utf-8 -*-
from typing import Union, Callable, Tuple, List
from cpu.core.types import Number, Tensor1D, Tensor2D, Tensor1D_or_3D
import cpu.core.math as math
from cpu.core.common import diap, at
import numpy as np


def trapz(a: Tensor1D_or_3D, lower: int, upper: int,
          dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
    return math.sum_(diap(a, lower + 1, upper) * diap(dh, lower + 1, upper), axis=-1) + \
           (at(a, lower) * at(dh, lower) +
            at(a, upper) * at(dh, upper)) / 2.


def simpson(a: Tensor1D_or_3D, lower: int, upper: int,
            dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
    return (at(a, lower) * at(dh, lower) +
            at(a, upper) * at(dh, upper) +
            4 * math.sum_(diap(a, lower + 1, upper, 2) *
                          diap(dh, lower + 1, upper, 2), axis=-1) +
            2 * math.sum_(diap(a, lower + 2, upper, 2) *
                          diap(dh, lower + 2, upper, 2), axis=-1)) / 3.


def boole(a: Tensor1D_or_3D, lower: int, upper: int,
          dh: Union[float, Tensor1D]) -> Union[Number, Tensor2D]:
    return (14 * (at(a, lower) * at(dh, lower) +
                  at(a, upper) * at(dh, upper)) +
            64 * math.sum_(diap(a, lower + 1, upper, 2) *
                           diap(dh, lower + 1, upper, 2), axis=-1) +
            24 * math.sum_(diap(a, lower + 2, upper, 4) *
                           diap(dh, lower + 2, upper, 4), axis=-1) +
            28 * math.sum_(diap(a, lower + 4, upper, 4) *
                           diap(dh, lower + 4, upper, 4), axis=-1)) / 45.


def limits(a: Tensor1D_or_3D, lower: int, upper: int,
           dh: Union[float, Tensor1D], method='trapz',
           theta: float = 0., px: float = 50., incline: Union[str, None] = 'left',
           boundaries: bool = False) -> Union[Number, Tensor2D, Tuple[Union[Number, Tensor2D],
                                                                      Union[List[Tuple[int, int]], None]]]:
    if np.isclose(theta, 0.):
        if method.lower() == 'trapz':
            a = trapz(a, lower, upper, dh)
        elif method.lower() == 'simpson':
            a = simpson(a, lower, upper, dh)
        else:  # boole
            a = boole(a, lower, upper, dh)

        if boundaries:
            return a, None
        return a

    rank = math.rank(a)
    if rank == 1:
        a = limits(a, lower, upper, dh, method) / math.cos(theta)

        if boundaries:
            return a, None
        return a

    elif rank == 3:
        Ix, Iy, Iz = a.shape

        if isinstance(dh, float) or math.rank(dh) == 0:
            py = Iz * dh
        elif math.rank(dh) == 1:
            py = math.sum_(dh)
        else:
            raise RuntimeError('wrong rank')

        dx = math.tan(theta) * py    # Определим смещение по Ox в км
        N = Ix / px                # Определим, сколько узлов приходится на 1 км
        di = dx * N                # Определим смещение по Ox в узлах
        if di >= Ix:
            raise RuntimeError('too big angle for such an array')

        Delta = int(Ix - di)
        b = math.zeros([Delta, Iy, Iz])

        START, STOP = [], []
        if incline == 'left':
            for n in range(0, Iz, 1):
                p = n / (Iz-1)
                start = int(di - di * p)
                stop = start + Delta
                b[:, :, n] = a[start:stop, :, n]
                START.append(start)
                STOP.append(stop)
        else:
            for n in range(0, Iz, 1):
                p = n / (Iz - 1)
                start = int(0 + di * p)
                stop = start + Delta
                b[:, :, n] = a[start:stop, :, n]
                START.append(start)
                STOP.append(stop)

        a = limits(b, lower, upper, dh, method)
        if boundaries:
            return a, list(zip(START, STOP))
        return a

    raise RuntimeError('wrong rank. Only 1D- or 3D-arrays')


def full(a: Tensor1D_or_3D, dh: Union[float, Tensor1D], method='trapz',
         theta: float = 0., px: float = 50., incline: str = 'left') -> Union[Number, Tensor2D]:
    return limits(a, 0, math.len_(a) - 1, dh, method, theta, px, incline)


def callable_f(f: Callable, lower: int, upper: int,
               dh: Union[float, Tensor1D], method='trapz',
               theta: float = 0., px: float = 50., incline: str = 'left',
               boundaries: bool = False) -> Union[Number, Tensor2D, Tuple[Union[Number, Tensor2D],
                                                                          Union[List[Tuple[int, int]], None]]]:
    a = math.as_tensor([f(i) for i in range(lower, upper + 1, 1)])
    if math.rank(a) == 3:
        a = math.transpose(a, axes=[1, 2, 0])
    return limits(a, lower, upper, dh, method, theta, px, incline, boundaries=boundaries)
