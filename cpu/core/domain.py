# -*- coding: utf-8 -*-
from typing import Tuple


class Domain3D:
    def __init__(self, kilometers: Tuple[float, float, float] = (50., 50., 10.),
                 nodes: Tuple[int, int, int] = (300, 300, 500)):
        """
        3D расчетная область

        :param kilometers: размеры по осям Ox, Oy и Oz в километрах
        :param nodes: кол-во узлов по соответствующим осям
        """
        self.PX, self.PY, self.PZ = kilometers
        self.kilometers = kilometers
        self.Nx, self.Ny, self.Nz = nodes
        self.nodes = nodes

    def x(self, i: int) -> float:
        return i * self.PX / (self.Nx - 1)

    def y(self, j: int) -> float:
        return j * self.PY / (self.Ny - 1)

    def z(self, k: int) -> float:
        return k * self.PZ / (self.Nz - 1)

    def i(self, x: float) -> int:
        return int(x / self.PX * (self.Nx - 1))

    def j(self, y: float) -> int:
        return int(y / self.PY * (self.Ny - 1))

    def k(self, z: float) -> int:
        return int(z / self.PZ * (self.Nz - 1))

    @property
    def dx(self) -> float:
        return self.x(1)

    @property
    def dy(self) -> float:
        return self.y(1)

    @property
    def dz(self) -> float:
        return self.z(1)

    @property
    def dh(self) -> float:
        return self.z(1)


class Column3D(Domain3D):
    def __init__(self, kilometers_z: float = 10., nodes_z: int = 500):
        super().__init__(kilometers=(1., 1., kilometers_z), nodes=(1, 1,  nodes_z))
