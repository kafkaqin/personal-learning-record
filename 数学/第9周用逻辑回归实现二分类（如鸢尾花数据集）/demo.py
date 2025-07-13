import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

iris = load_iris()
X = iris.data[iris.target!=2]
y = iris.target[iris.target!=2]

scaler = StandardScaler()
X = scaler.fit_transform(X)

y = y.reshape(-1, 1)

print("数据的形状: ",X.shape,y.shape)

class LogisticRegression:
    def __init__(self, input_dim):
        self.params = {
            'w':np.random.randn(input_dim,1)* 0.01,
            'b': np.zeros((1,))
        }

    def forward(self,X):
        z = X @ self.params['w'] + self.params['b']
        return self.sigmoid(z)

    def sigmoid(self,X):
        return 1/(1+np.exp(-X))

    def compute_loss(self, y_pred,y_true):
        epsilon = 1e-15
        loss = -np.mean(y_true * np.log(y_pred+epsilon) + (1-y_true) * np.log(1-y_pred+epsilon))
        return loss

    def backward(self,X, y_pred,y_true):
        N = X.shape[0]
        dz = y_pred - y_true
        dw = X.T @ dz /N
        db = np.sum(dz)/N
        return {"w":dw, 'b':db}

class AdamOptimizer:
    def __init__(self,params, lr=1e-3, betas=(0.9,0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0
        for key in params:
            self.m[key] = np.zeros_like(params[key])
            self.v[key] = np.zeros_like(params[key])
    def step(self,grads):
        self.t += 1
        for key in self.params:
            grad = grads[key]

            # 确保梯度与参数形状一致
            if self.params[key].shape != grad.shape:
                grad = grad.reshape(self.params[key].shape)

            self.m[key] = self.beta1 * self.m[key] +(1-self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] +(1-self.beta2) * (grad**2)

            m_hat = self.m[key] / (1-self.beta1**self.t)
            v_hat = self.v[key] / (1-self.beta2**self.t)
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    def zero_grad(self):
        pass


class LinearRegressionWithL2:
    def __init__(self,input_dim,l2_lambda=0.01):
        self.params = {
            'w': np.random.randn(input_dim,1) * 0.01,
            'b': np.zeros((1,))
        }
        self.l2_lambda = l2_lambda

    def forward(self,X):
         w,b = self.params['w'],self.params['b']
         return X @ w + b
    def compute_loss(self, y_pred, y_true):
        N = y_true.shape[0]
        mse_loss = 0.5 / N * np.sum((y_pred - y_true)**2)
        l2_penalty = 0.5 * self.l2_lambda * np.sum(self.params['w']**2)
        total_loss = mse_loss + l2_penalty
        return total_loss
    def backward(self,X, y_pred, y_true):
        N = X.shape[0]
        dw = (1.0/N) * X.T @(y_pred-y_true) + self.l2_lambda * self.params['w']
        db = (1.0 / N) * np.sum(y_pred-y_true,axis=0)
        return {'w':dw,'b':db}

model = LogisticRegression(input_dim=X.shape[1])
optimizer = AdamOptimizer(model.params, lr=0.1)

epochs = 300
for epoch in range(epochs):
    y_pred = model.forward(X)
    loss = model.compute_loss(y_pred,y_true=y)
    grads = model.backward(X,y_pred,y)
    optimizer.step(grads)
    if (epoch+1) % 50 == 0:
        acc = np.mean((y_pred > 0.5)==y)
        print(f"Epoch [{epoch+1}/{epochs}],Loss: {loss:.4f},Accuracy: {acc:.4f}")

y_pred_final = model.forward(X)
accuracy = np.mean((y_pred_final > 0.5) == y)
print("\n最终训练准确率:",accuracy)
