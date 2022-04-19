#  -*- coding: utf-8 -*-
from typing import Union
import numpy as np


Number = Union[float, complex]
TensorLike = Union[np.ndarray]
Tensor1D = Tensor2D = Tensor3D = Tensor1D_or_3D = TensorLike

cpu_float = np.float32
