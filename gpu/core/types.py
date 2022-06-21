#  -*- coding: utf-8 -*-
from typing import Union
import numpy as np
import tensorflow as tf


Number = Union[float, complex]
TensorLike = Union[np.ndarray, tf.Tensor, tf.Variable]
Tensor1D = Tensor2D = Tensor3D = Tensor1D_or_2D = Tensor1D_or_3D = TensorLike

cpu_float = np.float32
gpu_float = tf.float32
