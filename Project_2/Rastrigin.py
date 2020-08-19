import numpy as np
from math import *


def rastrigin(x, y):
    return 20 + x ** 2 + y ** 2 - 10 * cos(2 * pi * x) - 10 * cos(2 * pi * y)


def rastrigin_gradient(x, y):
    [grad_x, grad_y] = [2 * (x + 10 * pi * sin(2 * pi * x)), 2 * (y + 10 * pi * sin(2 * pi * y))]
    return np.array([grad_x, grad_y])


def rastrigin_hessian(x, y):
    [[h11, h12], [h21, h22]] = [[40 * (pi ** 2) * cos(2 * pi * x) + 2, 0], [0, 40 * (pi ** 2) * cos(2 * pi * x) + 2]]
    return [[h11, h12], [h21, h22]]
