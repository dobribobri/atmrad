
from typing import Tuple


class CylinderCloud:
    def __init__(self, _center: Tuple[float, float, float],
                 _rx: float, _ry: float, _height: float):
        self.x, self.y, self.z = _center
        self.rx, self.ry, self.height = _rx, _ry, _height

    def includesQ(self, _coords: Tuple[float, float, float]):
        _x, _y, _z = _coords
        return ((_x - self.x) * (_x - self.x) / (self.rx * self.rx) +
                (_y - self.y) * (_y - self.y) / (self.ry * self.ry) <= 1) and \
               (self.z <= _z) and (_z <= self.z + self.height)

    def belongsQ(self, _sizes: Tuple[float, float, float]):
        PX, PY, PZ = _sizes
        return ((0 <= self.x - self.rx) and (self.x + self.rx <= PX) and
                (0 <= self.y - self.ry) and (self.y + self.ry <= PY) and
                (self.z >= 0) and (self.z + self.height <= PZ))

    def disjointQ(self, _cloud: 'CylinderCloud'):
        return ((self.x - self.rx <= _cloud.x - _cloud.rx) and (self.x + self.rx <= _cloud.x - _cloud.rx)) or \
               ((self.x - self.rx >= _cloud.x + _cloud.rx) and (self.x + self.rx >= _cloud.x + _cloud.rx)) or \
               ((self.y - self.ry <= _cloud.y - _cloud.ry) and (self.y + self.ry <= _cloud.y - _cloud.ry)) or \
               ((self.y - self.ry >= _cloud.y + _cloud.ry) and (self.y + self.ry >= _cloud.y + _cloud.ry))

    @property
    def bottom(self):
        return self.z

    @property
    def top(self):
        return self.z + self.height
