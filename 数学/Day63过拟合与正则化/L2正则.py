import numpy as np
np.random.seed(42)

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

if __name__ == '__main__':
    np.random.seed(42)
    N,D = 100,5
    X = np.random.randn(N,D)
    true_w = np.array([[1.0],[-2.0],[0.5],[0.0],[0.0]])
    y_true = X @ true_w + np.random.randn(N,1)*0.1

    model = LinearRegressionWithL2(D,l2_lambda=0.1)
    optimizer = AdamOptimizer(model.params, lr=1e-2)
    for epoch in range(200):
        y_pred = model.forward(X)
        loss = model.compute_loss(y_pred,y_true)
        grads = model.backward(X,y_pred,y_true)
        optimizer.step(grads)
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/200],Loss: {loss:.6f}")
    print("\n训练后的权重:")
    print(model.params['w'])
    print("\n真实权重:")
    print(true_w)
