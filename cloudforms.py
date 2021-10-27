# -*- coding: utf-8 -*-
from typing import Tuple
# from pythonlangutil.overload import Overload, signature
# from domain import Domain3D


class CylinderCloud:
    def __init__(self, center: Tuple[float, float, float],
                 rx: float, ry: float, height: float):
        """
        Облако цилиндрической формы

        :param center: координаты центра эллипса в основании цилиндра, км
        :param rx: длина x-полуоси эллипса в основании, км
        :param ry: длина y-полуоси эллипса в основании, км
        :param height: высота цилиндра, км
        """
        self.x, self.y, self.z = center
        self.rx, self.ry, self.height = rx, ry, height

    def includesQ(self, coords: Tuple[float, float, float]) -> bool:
        """
        Проверить, лежит ли точка с заданными координатами (coords) внутри цилиндра (self)

        :param coords: координаты, км
        :return: True/False
        """
        x, y, z = coords
        return ((x - self.x) * (x - self.x) / (self.rx * self.rx) +
                (y - self.y) * (y - self.y) / (self.ry * self.ry) <= 1) and \
               (self.z <= z) and (z <= self.z + self.height)

    # @Overload
    # @signature('tuple')
    def belongsQ(self, sizes: Tuple[float, float, float]) -> bool:
        """
        Проверить, лежит ли облако (self) целиком внутри параллелепипеда с размерами sizes

        :param sizes: размеры параллелепипеда, км
        :return: True/False
        """
        PX, PY, PZ = sizes
        return ((0 <= self.x - self.rx) and (self.x + self.rx <= PX) and
                (0 <= self.y - self.ry) and (self.y + self.ry <= PY) and
                (self.z >= 0) and (self.z + self.height <= PZ))

    # @belongsQ.overload
    # @signature('Domain3D')
    # def belongsQ(self, domain: 'Domain3D') -> bool:
    #     """
    #     Проверить, лежит ли облако (self) целиком внутри расчетной облачности Domain3D
    #
    #     :param domain: объект Domain3D
    #     :return: True/False
    #     """
    #     return self.belongsQ((domain.PX, domain.PY, domain.PZ))

    def disjointQ(self, cloud: 'CylinderCloud') -> bool:
        """
        Проверить, пересекается ли облако (self) с другим облаком (cloud) по осям Ox или Oy

        :param cloud: объект CylinderCloud
        :return: True/False
        """
        return ((self.x - self.rx <= cloud.x - cloud.rx) and (self.x + self.rx <= cloud.x - cloud.rx)) or \
               ((self.x - self.rx >= cloud.x + cloud.rx) and (self.x + self.rx >= cloud.x + cloud.rx)) or \
               ((self.y - self.ry <= cloud.y - cloud.ry) and (self.y + self.ry <= cloud.y - cloud.ry)) or \
               ((self.y - self.ry >= cloud.y + cloud.ry) and (self.y + self.ry >= cloud.y + cloud.ry))

    @property
    def bottom(self) -> float:
        """
        :return: Высота основания цилиндрического облака, км
        """
        return self.z

    @property
    def top(self) -> float:
        """
        :return: Высота верхней границы облака, км
        """
        return self.z + self.height
