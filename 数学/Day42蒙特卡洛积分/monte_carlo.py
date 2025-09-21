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


import numpy as np
def estimate_pi(N):
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    inside = (x**2+y**2) <=1
    area = 4 *np.mean(inside)
    return area

if __name__ == '__main__':
    print(estimate_pi(10000000))