# -*- coding: utf-8 -*-
from typing import Tuple


class CylinderCloud:
    def __init__(self, center: Tuple[float, float, float],
                 rx: float, ry: float, height: float):
        """
        Облако цилиндрической формы

        :param center: координаты центра эллипса в основании цилиндра (км)
        :param rx: длина x-полуоси эллипса в основании (км)
        :param ry: длина y-полуоси эллипса в основании (км)
        :param height: высота цилиндра (км)
        """
        self.x, self.y, self.z = center
        self.rx, self.ry, self.height = rx, ry, height

    def includes_q(self, cords: Tuple[float, float, float]) -> bool:
        """
        Проверить, лежит ли точка с заданными координатами (cords) внутри цилиндра (self)

        :param cords: координаты, км
        :return: True/False
        """
        x, y, z = cords
        return ((x - self.x) * (x - self.x) / (self.rx * self.rx) +
                (y - self.y) * (y - self.y) / (self.ry * self.ry) <= 1) and \
               (self.z <= z) and (z <= self.z + self.height)

    def belongs_q(self, sizes: Tuple[float, float, float]) -> bool:
        """
        Проверить, лежит ли облако (self) целиком внутри параллелепипеда с размерами sizes

        :param sizes: размеры параллелепипеда (км)
        :return: True/False
        """
        px, py, pz = sizes
        return ((0 <= self.x - self.rx) and (self.x + self.rx <= px) and
                (0 <= self.y - self.ry) and (self.y + self.ry <= py) and
                (self.z >= 0) and (self.z + self.height <= pz))

    def disjoint_q(self, cloud: 'CylinderCloud') -> bool:
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
