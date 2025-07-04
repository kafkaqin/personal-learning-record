import numpy as np
import matplotlib.pyplot as plt

def integrand(x):
    return np.exp(-x**2)

a, b = 0 ,1
N = 1000000
x_sample = np.random.uniform(a, b, N)
f_values = integrand(x_sample)
intervals_estimate = (b-a)*np.mean(f_values)
print(intervals_estimate)

cumulative_mean = np.cumsum(f_values) /np.arange(1, N+ 1)
integral_convergence = (b -a) * cumulative_mean
print(integral_convergence)
