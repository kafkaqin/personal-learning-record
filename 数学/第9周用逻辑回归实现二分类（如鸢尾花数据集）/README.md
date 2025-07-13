非常好！我们可以使用 **逻辑回归（Logistic Regression）** 来实现一个简单的二分类任务。我们将使用 **鸢尾花数据集（Iris）**，从中提取两类（例如：`setosa` 和 `versicolor`），并使用 NumPy 手动实现前向传播、损失函数（交叉熵）、梯度计算和参数更新。

---

## 🎯 任务目标

- 使用 **逻辑回归** 实现 **二分类**
- 数据集：**Iris 鸢尾花数据集**（取两个类别）
- 损失函数：**二元交叉熵（Binary Cross Entropy）**
- 优化器：使用你之前写的 **AdamOptimizer**

---

## ✅ 步骤概览

1. 加载 Iris 数据并做预处理（标准化 + 二分类）
2. 定义逻辑回归模型
3. 前向传播（Sigmoid 函数）
4. 计算损失（BCE Loss）
5. 反向传播求梯度
6. 使用 Adam 优化器更新参数
7. 进行训练和评估

---

## 💻 完整代码如下：

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

# -------------------------------
# 1. 加载并预处理数据
# -------------------------------
iris = load_iris()
X = iris.data[iris.target != 2]  # 只保留前两类（0: setosa, 1: versicolor）
y = iris.target[iris.target != 2]

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 转换为二维数组 (N, 1)
y = y.reshape(-1, 1)

print("数据形状：", X.shape, y.shape)

# -------------------------------
# 2. 定义逻辑回归模型
# -------------------------------
class LogisticRegression:
    def __init__(self, input_dim):
        self.params = {
            'w': np.random.randn(input_dim, 1) * 0.01,
            'b': np.zeros((1, ))  # shape (1,)
        }

    def forward(self, X):
        z = X @ self.params['w'] + self.params['b']
        return self.sigmoid(z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_pred, y_true):
        # 二元交叉熵损失函数
        epsilon = 1e-15  # 防止 log(0)
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    def backward(self, X, y_pred, y_true):
        N = X.shape[0]
        dz = y_pred - y_true
        dw = X.T @ dz / N
        db = np.sum(dz) / N
        return {'w': dw, 'b': db}

# -------------------------------
# 3. 实例化模型和优化器
# -------------------------------
model = LogisticRegression(input_dim=X.shape[1])
optimizer = AdamOptimizer(model.params, lr=0.1)

# -------------------------------
# 4. 训练循环
# -------------------------------
epochs = 300
for epoch in range(epochs):
    y_pred = model.forward(X)
    loss = model.compute_loss(y_pred, y_true=y)
    grads = model.backward(X, y_pred, y)

    optimizer.step(grads)

    if (epoch + 1) % 50 == 0:
        acc = np.mean((y_pred > 0.5) == y)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# -------------------------------
# 5. 最终预测准确率
# -------------------------------
y_pred_final = model.forward(X)
accuracy = np.mean((y_pred_final > 0.5) == y)
print("\n最终训练准确率：", accuracy)
```

---

## 📈 示例输出（可能略有不同）：

```
数据形状： (100, 4) (100, 1)
Epoch [50/300], Loss: 0.2231, Accuracy: 0.9200
Epoch [100/300], Loss: 0.1023, Accuracy: 0.9800
Epoch [150/300], Loss: 0.0532, Accuracy: 1.0000
Epoch [200/300], Loss: 0.0312, Accuracy: 1.0000
...
最终训练准确率： 1.0
```

---

## 📌 总结

我们完成了以下内容：

✅ 使用逻辑回归实现了对 Iris 数据集中两个类别的分类  
✅ 使用 Sigmoid 激活函数和 BCE 损失函数  
✅ 手动实现反向传播计算梯度  
✅ 使用你之前写的 `AdamOptimizer` 更新参数  
✅ 达到了接近 100% 的训练准确率

---

## 🧩 后续扩展建议

你可以继续改进这个模型，比如：

- 添加验证集进行泛化评估
- 使用 mini-batch 训练
- 绘制损失曲线
- 可视化决策边界（仅限二维输入）
