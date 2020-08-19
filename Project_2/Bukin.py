import numpy as np
from math import pi, sqrt, sin, cos
from numpy import abs, square


def bukin(x, y):
    return 100 * sqrt(abs(y - 0.01 * square(x))) + 0.01 * abs(x + 10)


def bukin_gradient(x, y):
    if x <= -10:
        if y > (x ** 2)/100:
            [grad_x, grad_y] = [0.01 - x/sqrt(y - 0.01 * x ** 2), 50/sqrt(y - 0.01 * x ** 2)]
        else:
            [grad_x, grad_y] = [0.01 + x / sqrt(-y + 0.01 * x ** 2), -50 / sqrt(-y + 0.01 * x ** 2)]
    else:
        if y > (x ** 2)/100:
            [grad_x, grad_y] = [-0.01 - x/sqrt(y - 0.01 * x ** 2), 50/sqrt(y - 0.01 * x ** 2)]
        else:
            [grad_x, grad_y] = [-0.01 + x / sqrt(-y + 0.01 * x ** 2), -50 / sqrt(-y + 0.01 * x ** 2)]

    return np.array([grad_x, grad_y])
