#  -*- coding: utf-8 -*-
from typing import Union, List, Tuple, Callable
from multiprocessing import Manager, Process
from cpu.core.types import cpu_float
import numpy as np


def do(processes: list, n_workers: int) -> None:
    for i in range(0, len(processes), n_workers):
        for j in range(i, i + n_workers):
            if j < len(processes):
                processes[j].start()
        for j in range(i, i + n_workers):
            if j < len(processes):
                processes[j].join()


def parallel(enumerable: Union[np.ndarray, List[float]],
             func: Callable, args: Union[Tuple, List],
             n_workers: int) -> np.ndarray:
    if not n_workers:
        n_workers = len(enumerable)
    with Manager() as manager:
        out = manager.list()
        processes = []
        for i, f in enumerate(enumerable):
            p = Process(target=out.append, args=((i, func(f, *args)),))
            processes.append(p)
        do(processes, n_workers)
        out = list(out)
    return np.asarray([val for _, val in sorted(out, key=lambda item: item[0])], dtype=cpu_float)
