import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.random.randn(100, 1) * 2
y = 4+3*X+np.random.randn(100,1)

w = 0.0
b = 0.0
learning_rate = 0.1
n_iterations = 1000
m = len(X)

for iteration in range(n_iterations):
    y_pred = w * X + b
    error = y_pred - y

    gradient_w = (2/m)*np.dot(X.T,error).item()
    gradient_b = (2/m)*np.sum(error)

    w -=learning_rate * gradient_w
    b -=learning_rate * gradient_b

    if iteration % 100 == 0:
        loss = np.mean(error*2)
        print('Iteration {}, Loss = {:.4f}'.format(iteration, loss))

print("\n最终模型参数:")
print(f"斜率 w = {w:.4f}, 截距 b = {b:.4f}")

plt.scatter(X,y,color='blue',label='数据点')
plt.plot(X,w * X+b,color='red',label="拟合直线")
plt.xlabel('x')
plt.ylabel('y')
plt.title("线性回归拟合")
plt.legend()
plt.grid(True)
plt.savefig('demo11.png')