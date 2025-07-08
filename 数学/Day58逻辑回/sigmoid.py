import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

np.random.seed(42)
X_class0 = np.random.randn(50, 2) + [2,2]
X_class1 = np.random.randn(50, 2) + [-2,-2]
X = np.vstack((X_class0, X_class1))

y = np.array([0]*50 + [1]*50).reshape(-1,1)
X_bias = np.hstack((X, np.ones((X.shape[0], 1))))


weights = np.random.randn(3,1)
learning_rate = 0.1
n_iterations = 1000
for i in range(n_iterations):
    z = X_bias @ weights
    y_pred = sigmoid(z)

    if i % 200 == 0:
        loss = -np.mean(y * np.log(y_pred+1e-15)+(1-y)*np.log(1-y_pred+1e-15))
        print(f"Iteration {i} loss = {loss:.4f}")

gradient = (y_pred-y) * X_bias
gradient = np.mean(gradient, axis=0).reshape(-1,1)
weights -= learning_rate * gradient


def predict(X,weights):
    z = X @ weights
    prob = sigmoid(z)
    return (prob > 0.5).astype(int)

test_points = np.array([[3,3],[-3,-3]])
test_points_bias = np.hstack((test_points, np.ones((test_points.shape[0], 1))))
predictions = predict(test_points_bias, weights)

print("测试点预测结果:")
for point,label in zip(test_points, predictions):
    print(f"点 {point} 被预测为类别 {label.item()}")


x_vals = np.linspace(-5, 5, 100)
y_vals = -(weights[0] * x_vals + weights[2]) / weights[1]

plt.scatter(X_class0[:, 0], X_class0[:, 1], label="Class 0", color='blue')
plt.scatter(X_class1[:, 0], X_class1[:, 1], label="Class 1", color='red')
plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')
plt.title('Sigmoid 分类器的决策边界')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.savefig('sigmoid.png')