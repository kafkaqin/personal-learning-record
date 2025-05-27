import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.rand(100, 1) * 10

y = 2.5 * X.squeeze() + 1.2+np.random.randn(100)*2
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

coefficients, residuals, rank, singular_values=np.linalg.lstsq(X_b, y)
print("模型参数（截距和斜率）:", coefficients)
y_pred = X_b @coefficients

plt.scatter(X, y, color='blue', label='原始数据')
plt.plot(X, y_pred, color='red', label='拟合直线')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('NumPy 实现线性回归')
plt.grid(True)
plt.savefig("dss.png")
