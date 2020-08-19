import numpy as np
from math import cos, sqrt, sin, exp, pi, e
from numpy import square


def ackley(x, y):
    return -20 * exp(-0.2 * sqrt((x ** 2 + y ** 2)/2)) - exp(cos(2 * pi * x) + cos(2 * pi * y)) + e + 20


def ackley_gradient(x, y):
    [grad_x, grad_y] = [(2.82843 * x * exp((-0.141421 * sqrt(x ** 2 + y ** 2))))/sqrt(x ** 2 + y ** 2) + 2 * pi
                        * sin(2 * pi * x) * exp((cos(2 * pi * x) + cos(2 * pi * y))),
                        (2.82843 * y * exp((-0.141421 * sqrt(x ** 2 + y ** 2))))/sqrt(x ** 2 + y ** 2) + 2 * pi
                        * sin(2 * pi * y) * exp((cos(2 * pi * x) + cos(2 * pi * y)))]
    return np.array([grad_x, grad_y])
