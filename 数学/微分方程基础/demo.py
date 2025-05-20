from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

def ode_function(t, y):
    return -2*t*y

if __name__ == '__main__':
    y0=[1]
    t_span = (0,5)
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    sol = solve_ivp(ode_function, t_span, y0,method="RK45", t_eval=t_eval)


    plt.plot(sol.t,sol.y[0],label='Numerical Solution')
    plt.plot(sol.t,np.exp(-sol.t**2),'r--',label='Analytical Solution')
    plt.title('Solution of the ODE')
    plt.xlabel('Time t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.savefig('ODE.png')