import numpy as np
from scipy.optimize import minimize

def objective(x):
    return (x[0] -3)**2 + (x[1] - 4)**2
def gradient(x):
    return np.array([2*(x[0]-3) , 2*(x[1] - 4)])

def hessian(x):
    return np.array([[2,0],[0,2]])

def gradient_descent(starting_point,learning_rate=0.1,iterations=100):
    x = starting_point
    path = []
    for i in range(iterations):
        x = x -learning_rate * gradient(x)
        path.append(x.copy())
    return np.array(path)

result_newton = minimize(objective,np.array([0,0]),method='Newton-CG',jac=gradient ,hess=hessian)
starting_point = np.array([0,0])
path_gd = gradient_descent(starting_point,learning_rate=0.1,iterations=100)

print("梯度下降路径：")
print(path_gd)
print("\n牛顿法结果:")
print(result_newton.x)
