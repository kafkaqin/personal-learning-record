import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def dy_dt(t, y):
    return -2 * y

y0 = [1.0]
t_span = (0,5)
t_eval = np.linspace(0,5, 100)
sol = solve_ivp(dy_dt, t_span, y0, t_eval=t_eval,method='RK45')
print(sol)
plt.figure(figsize=(8,5))
plt.plot(sol.t,sol.y[0],'b-',label='数接值')
plt.plot(t_eval, np.exp(-2 * t_eval), 'r--', label='解析解 $e^{-2t}$')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('一阶 ODE: $dy/dt = -2y$')
plt.legend()
plt.grid(True)
plt.savefig('demo-1.png')

def harmonic_oscillator(t, y):
    y1,y2 = y
    dy1_dt = y2
    dy2_dt = -y1
    return [dy1_dt, dy2_dt]

y0 = [1.0,0.0]
t_span = (0,10)
t_eval = np.linspace(0,10,200)
sol = solve_ivp(harmonic_oscillator,t_span,y0,t_eval=t_eval,method='RK45')


def lotka_volterra(t, z):
    x , y = z
    dx_dt = x * (1-y)
    dy_dt = y * (x-1)
    return [dx_dt, dy_dt]

z0 = [2.0,1.0]
sol = solve_ivp(lotka_volterra,(0,20),z0,t_eval=t_eval,method='RK45')
print(sol)