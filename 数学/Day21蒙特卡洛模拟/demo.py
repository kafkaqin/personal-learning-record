import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
N = 1000
x = np.random.uniform(-1,1,N)
y = np.random.uniform(-1,1,N)
inside = x**2 + y**2 <=1
pi_estimate = 4 * np.mean(inside)

print(f"使用{N:,}个随机点")
print(f"pi的估计值{pi_estimate:.6f}")
print(f"pi的真实值{np.pi:.6f}")
print(f"误差:{pi_estimate-np.pi:.6f}")

N = 10000

x = np.random.uniform(0,1,N)
f_x = x**2
integral_estimate = np.mean(f_x)
true_value = 1/3


N = 50000
a ,b = 0,2

x = np.random.uniform(a,b,N)
f_x = np.sin(x)
integral_estimate = (b-a)*np.mean(f_x)
true_value = -np.cos(b)+np.cos(0)

