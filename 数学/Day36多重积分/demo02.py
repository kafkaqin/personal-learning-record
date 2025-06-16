from scipy.integrate import dblquad
import numpy as np

def integrand(y,x):
    return x**2 + y**2


result,err = dblquad(func=integrand,a=0,b=1,gfun=lambda x:-np.sqrt(1-x**2),hfun=lambda x:np.sqrt(1-x**2))
print(result)
print(err)