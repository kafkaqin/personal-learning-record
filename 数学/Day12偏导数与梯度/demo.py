import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    return x**2+2*y**2+2*x*y

def gradient(x,y):
    df_dx = 2*x+2*y
    df_dy = 4*y+2*y
    return np.array([df_dx,df_dy])

x0,y0 = 1.0,1.0
grad = gradient(x0,y0)
print(grad)


def numerical_gradient(func,variables,eps=1e-6):
    grad = np.zeros_like(variables)
    for i in range(len(variables)):
        h = np.zeros_like(variables)
        h[i] = eps
        grad[i] = (func(*(variables+h))-func(*(variables-h)))/ (2*eps)
    return grad

point = np.array([1.0,1.0])
num_grad = numerical_gradient(f,point)
print(num_grad)


def gradient_descent(func,grad_func,start,learning_rate=0.1,max_iters=100,tol=1e-6):
    x = np.array(start,dtype=float)
    trajectory = [x.copy()]
    for i in range(max_iters):
        grad = grad_func(x[0],x[1])
        x_new = x - learning_rate*grad

        if np.linalg.norm(x_new-x) < tol:
            print(f'Converged at iteration {i+1}')
            break
        x = x_new
        trajectory.append(x.copy())

    return x,func(x[0],x[1]),np.array(trajectory)

start_point = [2.0,2.0]

min_point,min_value,trajectory = gradient_descent(f,gradient,start_point,learning_rate=0.1)
print(f"\n找到的最小值点: x = {min_point[0]:.6f}, y = {min_point[1]:.6f}")
print(f"最小值: f(x,y) = {min_value:.6f}")

# 创建网格用于画等高线
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=20, colors='black', alpha=0.5)
plt.clabel(contour, inline=True, fontsize=8)

# 填充颜色
plt.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)

# 绘制优化路径
plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2, markersize=4, label='梯度下降路径')
plt.scatter(0, 0, color='green', s=100, label='理论最小值 (0,0)')
plt.scatter(start_point[0], start_point[1], color='purple', s=100, label='起点')

plt.xlabel('x')
plt.ylabel('y')
plt.title('梯度下降法优化路径')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.colorbar(label='f(x,y)')
plt.savefig('demo11.png')