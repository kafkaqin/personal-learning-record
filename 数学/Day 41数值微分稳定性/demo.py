import numpy as np
import matplotlib.pyplot as plt

def exact_solution(t):
    return np.exp(-2*t)

def euler_step(f,t,y,h):
    return y+h*f(t,y)

def rk4_step(f,t,y,h):
    k1 = h*f(t,y)
    k2 = h*f(t+h/2,y+k1/2)
    k3 = h*f(t+h/2,y+k2/2)
    k4 = h*f(t+h,y+k3)
    return y+(k1+2*k2+2*k3+k4)/6

def dydt(t,y):
    return -2*y

t0 = 0.0
y0 = 1.0
T = 5.0
h = 0.2

t_values = np.arange(t0,T+h,h)
euler_values = []
rk4_values = []

y_euler = y0
y_rk4 = y0

for t in t_values:
    euler_values.append(y_euler)
    rk4_values.append(y_rk4)

    y_euler = euler_step(dydt,t,y_euler,h)
    y_rk4 = rk4_step(dydt,t,y_rk4,h)

exact_values = exact_solution(t_values)


plt.figure(figsize=(10,6))
plt.plot(t_values,euler_values,'k-',lw=2,label='Exact solution')
plt.plot(t_values,euler_values,'r--o',label='Euler method')
plt.plot(t_values,rk4_values,'b--s',label='rk4 method')

plt.title("Comparison of Euler and RK4 Methods")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.savefig("exact_solution.png")


from scipy.integrate import solve_ivp

sol_euler = solve_ivp(dydt, (t0, T), [y0], method='RK23', max_step=h)
sol_rk45 = solve_ivp(dydt, (t0, T), [y0], method='RK45', max_step=h)

plt.figure(figsize=(10, 6))
plt.plot(sol_euler.t, sol_euler.y[0], 'r--o', label='SciPy Euler-like (RK23)')
plt.plot(sol_rk45.t, sol_rk45.y[0], 'g--s', label='SciPy RK45')
plt.plot(t_values, exact_values, 'k-', lw=2, label='Exact Solution')
plt.legend()
plt.grid()
plt.savefig("exact_solution1.png")