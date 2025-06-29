import numpy as np
import matplotlib.pyplot as plt
num_samples = 100000
x = np.random.uniform(-1,1,num_samples)
y = np.random.uniform(-1,1,num_samples)

in_circle = x**2+y**2<=1
count_in_circle = np.sum(in_circle)

pi_estimate = 4 * count_in_circle/num_samples
print(f"估算 pi的值：{pi_estimate:.6f}")
