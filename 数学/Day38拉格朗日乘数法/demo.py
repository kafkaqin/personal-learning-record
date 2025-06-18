import numpy as np
from scipy.optimize import minimize

def objective(vars):
    return (vars[0]-1)**2 + (vars[1]-2.5)**2

def constraint1(vars):
    return 6-(vars[0]+2*vars[1])

def constraint2(vars):
    return vars[0]-2*vars[1]+2

initial_guess = [2,0]
cons = ({'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2})

bounds = ((0,None),(0,None))

solution = minimize(objective, initial_guess,method="SLSQP", constraints=cons, bounds=bounds)

if solution.success:
    print("优化成功:")
    print("x = ",solution.x[0])
    print("y = ",solution.x[1])
    print("最小值: ",solution.fun)
else:
    print("未能找到最优解")