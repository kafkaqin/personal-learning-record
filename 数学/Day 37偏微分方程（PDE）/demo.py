import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 1.0             # 空间长度
T = 0.5             # 总时间
alpha = 0.01        # 热扩散率
Nx = 50             # 空间网格数
Nt = 1000           # 时间步数
dx = L / (Nx - 1)
dt = T / Nt
nu = alpha * dt / dx**2

print(f"ν = {nu:.4f}, 稳定性要求 ν ≤ 0.5")

# 初始化网格
x = np.linspace(0, L, Nx)
u = np.zeros((Nx, Nt+1))

# 初始条件：三角形分布
u[:, 0] = np.sin(np.pi * x)

# 边界条件
u[0, :] = 0
u[-1, :] = 0

# 显式迭代
for n in range(Nt):
    for i in range(1, Nx-1):
        u[i, n+1] = u[i, n] + nu * (u[i+1, n] - 2*u[i, n] + u[i-1, n])

# 可视化
plt.figure(figsize=(10, 6))
for n in [0, 10, 50, 100, 200, 500, 1000]:
    plt.plot(x, u[:, n], label=f't={n*dt:.3f}s')
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("热传导方程的有限差分解")
plt.legend()
plt.grid(True)
plt.savefig("fdm.png")