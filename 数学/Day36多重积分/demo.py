from scipy.integrate import dblquad
import numpy as np

def integrand(y,x):
    return x**2 + y**2


result,err = dblquad(func=integrand,a=0,b=1,gfun=lambda x:0,hfun=lambda x:1)
print(result)
print(err)

import scipy.integrate as integrate
def integrand(y,x):
    return x**2+y**2

def y_lower(x):
    return 0
def y_upper(x):
    return (1-x**2)**0.05
result,error = integrate.dblquad(integrand,0,1, y_lower, y_upper)
print(result)
print(error)