import numpy as np
import matplotlib.pyplot as plt

def trapezoidal_rule(f,a,b,n):
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    y = f(x)
    integral = h*(0.5*y[0]+0.5*y[-1]+np.sum(y[1:-1]))
    return integral

def f(x):
    return np.sin(x)
a = 0
b = np.pi
n = 100
result = trapezoidal_rule(f,a,b,n)

print(result)

def plot_trapezoidal(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, 'b-', label=r'$f(x) = \sin(x)$')
    plt.fill_between(x_fine, y_fine, color='lightblue', alpha=0.5)

    for i in range(n):
        xs = [x[i], x[i+1]]
        ys = [y[i], y[i+1]]
        plt.fill_between(xs, ys, color='red', alpha=0.4, edgecolor='black')
        plt.plot(xs, ys, 'r--', linewidth=1)

    plt.scatter(x, y, color='red', zorder=5)
    plt.title(f'梯形法则 (n = {n} 个区间)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.savefig("trapezoidal_rule.png")

# 绘图（用较少的区间以便观察）
plot_trapezoidal(f, 0, np.pi, n=6)


n_values = [10, 20, 50, 100, 200]
errors = []

for n in n_values:
    result = trapezoidal_rule(f, 0, np.pi, n)
    error = abs(result - 2)
    errors.append(error)
    print(f"n={n:3d}, 积分={result:.6f}, 误差={error:.2e}")

# 可视化误差
plt.loglog(n_values, errors, 'o-', label='梯形法则误差')
plt.loglog(n_values, [1/n**2 for n in n_values], '--', label=r'$O(1/n^2)$', color='red')
plt.xlabel('n (区间数)')
plt.ylabel('误差')
plt.title('梯形法则误差收敛性')
plt.legend()
plt.grid(True)
plt.savefig("trapezoidal_rule1.png")