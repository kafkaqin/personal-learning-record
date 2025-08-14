import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def f(x):
    return x**2 + 5*np.sin(x)

x0 = 0.0

result = minimize(f, x0, method='BFGS')
print(result.x[0])
print(result.fun)
print(result.success)

def f_2d(vars):
    x,y = vars
    return (x-2)**2 + (y+1)**2


x0 = [0,0]

result2d = minimize(f_2d, x0, method='BFGS')
print(result2d.x)
print(result2d.fun)
print(result2d.success)


def f_bounded(x):
    return (x-2)**2

x0 = [0.0]

bounds = [(0,3)]

result_bound = minimize(f_bounded, x0, method='BFGS', bounds=bounds)
print(result_bound.x)
print(result_bound.fun)
print(result_bound.success)