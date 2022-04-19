#  -*- coding: utf-8 -*-
from typing import Union, Tuple, List
from cpu.core.types import Number, Tensor2D, Tensor1D_or_3D, TensorLike
import cpu.core.math as math


def diap(a: Union[Number, Tensor1D_or_3D], start: int, stop: int,
         step: int = 1) -> Union[Number, TensorLike]:
    rank = math.rank(a)
    if rank == 0:
        return a
    if rank == 1:
        return a[start:stop:step]
    if rank == 3:
        return a[:, :, start:stop:step]
    raise RuntimeError('wrong rank')


def at(a: Union[float, Tensor1D_or_3D], index: int) -> Union[Number, Tensor2D]:
    rank = math.rank(a)
    if rank == 0:
        return a
    if rank == 1:
        return a[index]
    if rank == 3:
        return a[:, :, index]
    raise RuntimeError('wrong rank')


def cx(a: Union[Number, TensorLike], boundaries_profile: List[Tuple[int, int]] = None, height: int = None):
    if not boundaries_profile:
        return a
    rank = math.rank(a)
    if rank in [0, 1]:
        return a
    if rank in [2, 3]:
        start, stop = boundaries_profile[height]
        return a[start:stop]
    raise RuntimeError('wrong rank')
