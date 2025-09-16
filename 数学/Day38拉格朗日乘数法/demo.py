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

 import numpy as np
 from scipy.optimize import minimize

 def objective(x):
     return x[0] * x[3] *(x[0] +x[1]+x[2])+x[2]

 def constraint1(x):
     return x[0]*x[1]*x[2]*x[3]-25
 def constraint2(x):
     return np.sum(x**2)-40

 cons = [
     {'type': 'ineq', 'fun': constraint1},
     {'type': 'eq', 'fun': constraint2},
 ]
 def constraint3(x):
     return np.diff(x)
 cons.append({'type':'ireq','fun':constraint3})
 bounds = [(1,5) for _ in range(4)]
 x0 = [1,5,5,1]
 solution = minimize(objective, x0, method='SLSQP', bounds=bounds,constraints=cons)
 print("是否收敛成功：",solution.success)
 if solution.success:
     print("最优解 x*:",np.round(solution.x,4))
     print("最优值 f(x*)",round(solution.fun,4))
     print("约束值：")
     print(f"x1*x2*x3*x4={constraint1(solution.x)+25:.4f} >=25")
     print(f" x1^2+x2^2+x3^2+x4^2 = {constraint2(solution.x)+40:.4f} == 40")
 else:
     print("求解失败")

 x_opt = solution.x
 print("===验证===")
 print(f"目标函数值：{objective(x_opt):.4f}")
 g1 = constraint1(x_opt)
 print(f"不等式约束 g1={g1:.4e} >=0 ?{g1 >=-1e-6}")
 h1 = constraint2(x_opt)
 print(f"等式约束 h1={h1:.4e }")