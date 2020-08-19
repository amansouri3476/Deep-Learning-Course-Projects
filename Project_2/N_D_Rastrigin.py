import numpy as np
from math import cos, pi, sin


def n_dim_rastrigin(x):
    sum_rastrigin = 20
    for i in range(len(x)):
        sum_rastrigin += x[i] ** 2 - 10 * cos(2 * pi * x[i])
    return sum_rastrigin


def n_dim_rastrigin_gradient(x):
    gradient = []
    for i in range(len(x)):
        gradient.append(2 * (x[i] + 10 * pi * sin(2 * pi * x[i])))
    return np.array(gradient)

