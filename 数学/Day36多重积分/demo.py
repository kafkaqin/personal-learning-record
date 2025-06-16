from scipy.integrate import dblquad
import numpy as np

def integrand(y,x):
    return x**2 + y**2


result,err = dblquad(func=integrand,a=0,b=1,gfun=lambda x:0,hfun=lambda x:1)
print(result)
print(err)