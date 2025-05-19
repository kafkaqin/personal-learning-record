import numpy as np

def f(x):
    return x[0]**2 + (x[1]-3)**2

def grad_f(x,eps=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad[i] = (f(x+dx)-f(x-dx))/(2*eps)
    return grad

def gradient_descent(starting_point,learning_rate=0.1,max_iter=1000,tol=1e-6):
    x = np.array(starting_point,dtype=float)
    for i in range(max_iter):
        grad = grad_f(x)
        step = learning_rate * grad
        if np.linalg.norm(step) < tol:
            print(f"收敛于第{i}次迭代")
            break
        x -=step
        if i % 100 == 0:
            print(f"迭代 {i}: x = {x}, f(x) = {f(x)}")
    return x ,f(x)


if __name__ == '__main__':
    x0 = [1.0, 1.0]
    minimum_point, minimum_value = gradient_descent(x0)
    print("极小值点",minimum_point)
    print("极小值",minimum_value)
