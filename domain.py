
from typing import Tuple, Union


class Domain:
    def __init__(self, kilometers: Tuple[float, float, float] = None,
                 nodes: Tuple[Union[float, None], Union[float, None], float] = None, **kwargs):
        if kilometers is None:
            kilometers = (50., 50., 10.)
        if nodes is None:
            nodes = (300, 300, 500)
        self.PX, self.PY, self.PZ = kilometers
        self.Nx, self.Ny, self.Nz = nodes
        if self.Nx is None:
            self.Nx = 1
        if self.Ny is None:
            self.Ny = 1
        self.cl_bottom = 1.5
        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def x(self, i: int) -> float:
        if not i:
            return 0.
        return i * self.PX / (self.Nx - 1)

    def y(self, j: int) -> float:
        if not j:
            return 0.
        return j * self.PY / (self.Ny - 1)

    def z(self, k: int) -> float:
        if not k:
            return 0.
        return k * self.PZ / (self.Nz - 1)

    def i(self, x: float) -> int:
        if x == 0.:
            return 0
        return int(x / self.PX * (self.Nx - 1))

    def j(self, y: float) -> int:
        if y == 0.:
            return 0
        return int(y / self.PY * (self.Ny - 1))

    def k(self, z: float) -> int:
        if z == 0.:
            return 0
        return int(z / self.PZ * (self.Nz - 1))
