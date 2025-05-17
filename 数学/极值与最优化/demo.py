from scipy.optimize import minimize
import numpy as np

def objective_function(x):
    return x[0]**2+(x[1] - 3)**2

x0 = [1,2]

result = minimize(objective_function, x0, method="L-BFGS-B")
print("成功:",result.success)
print("最终参数值",result.x)
print("目标函数",result.fun)



cons = ({'type': 'eq', 'fun': lambda x:x[0]+x[1]-1})
bnds = ((0, None), (0, None))
result = minimize(objective_function,x0,method="SLSQP", bounds=bnds, constraints=cons)
print("成功1:",result.success)
print("最终参数值1==",result.x)
print("目标函数==",result.fun)