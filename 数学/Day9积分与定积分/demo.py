import scipy.integrate as integrate

def f(x):
    return x**2

result,error = integrate.quad(f, 0,1)
print(result,error)

from sympy import symbols,integrate

x = symbols('x')

f = x ** 2
integrate_result = integrate(f,(x,0,1))
print(integrate_result)