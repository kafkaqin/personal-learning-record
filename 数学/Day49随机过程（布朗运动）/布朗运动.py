import numpy as np
import matplotlib.pyplot as plt
T = 1.0
N = 1000
dt = T/N
t= np.linspace(0,T,N)

positon = np.zeros(N)

dW = np.random.normal(0,np.sqrt(dt),N)
positon = np.cumsum(dW)

plt.figure(figsize=(10,6))
plt.plot(t,positon,label='布朗运动路径',lw=2)
plt.title("布朗运动路径模拟")
plt.xlabel("时间")
plt.ylabel("位置")
plt.legend()
plt.savefig("布朗运动路径模拟.png")

num_paths = 5
plt.figure(figsize=(10,6))

for i in range(num_paths):
    dW = np.random.normal(0,np.sqrt(dt),N)
    positon = np.cumsum(dW)
    plt.plot(t,positon,lw=1.5,alpha=0.7)

plt.title(f"{num_paths}条布朗运动路径模拟")
plt.xlabel("时间")
plt.ylabel("位置")
plt.savefig("2.png")