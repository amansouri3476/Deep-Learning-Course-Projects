import numpy as np
from numpy import sin, cos, square
from math import *


def levi(x, y):
    return square(sin(3 * pi * x)) + square((x - 1)) * (1 + square(sin(3 * pi * y))) + square(y - 1) \
           * (1 + square(2 * pi * y))


def levi_gradient(x, y):
    [grad_x, grad_y] = [3 * (x + pi * sin(6 * pi * x) - 1) - (x - 1) * cos(6 * pi * y), 3 * pi * square(x - 1) * sin(6 * pi * y) + 2 * pi * square(y - 1) * sin(4 * pi * y) - (y - 1) * (cos(4 * pi * y) - 3)]
    return np.array([grad_x, grad_y])


def levi_hessian(x, y):
    [[h11, h12], [h21, h22]] = [[-18 * (pi ** 2) * square(sin(3 * pi * x)) + 18 * (pi ** 2) * square(cos(3 * pi * x)) + 2 * (square(sin(3 * pi * y)) + 1), 12 * pi * (x - 1) * sin(3 * pi * y) * cos(3 * pi * y)], [12 * pi * (x - 1) * sin(3 * pi * y) * cos(3 * pi * y), -18 * (pi ** 2) * square(x - 1) * square(sin(3 * pi * y)) + 18 * (pi ** 2) * square(x - 1) * square(cos(3 * pi * y)) + 2 * (4 * (pi ** 2) * (y ** 2) + 1) + 8 * (pi ** 2) * square(y - 1) + 32 * (pi ** 2) * y * (y - 1)]]
    return [[h11, h12], [h21, h22]]
