
import numpy as np

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 模拟数据
num_samples = 100
input_size = 4
hidden_size = 10
output_size = 3

# 随机生成一些输入数据和 one-hot 编码的标签
X = np.random.randn(num_samples, input_size)
y_true = np.eye(output_size)[np.random.choice(output_size, num_samples)]

# 初始化参数
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# 定义激活函数
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数值稳定性处理
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 前向传播
def forward(X, W1, b1, W2, b2):
    # 第一层线性变换 + ReLU 激活
    z1 = X @ W1 + b1
    a1 = relu(z1)

    # 第二层线性变换 + Softmax 输出
    z2 = a1 @ W2 + b2
    y_pred = softmax(z2)

    return y_pred, a1, z1

# 执行前向传播
y_pred, a1, z1 = forward(X, W1, b1, W2, b2)

# 查看输出形状
print("输入 X 形状:", X.shape)
print("预测概率 y_pred 形状:", y_pred.shape)
print("隐藏层激活值 a1 形状:", a1.shape)


learning_rate = 0.001
epochs = 100
for epoch in range(epochs):
    y_pred, a1, z1 = forward(X, W1, b1, W2, b2)

    dy = y_pred - y_true
    dW2 = a1.T @ dy
    db2 = np.sum(dy, axis=0, keepdims=True)

    da1 = dy @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W2 -=learning_rate * dW2
    b2 -=learning_rate * db2
    W1 -=learning_rate * dW1
    b1 -= learning_rate * db1