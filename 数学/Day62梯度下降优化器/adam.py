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
            self.m[key] = self.beta1 * self.m[key] +(1-self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] +(1-self.beta2) * (grad**2)

            m_hat = self.m[key] / (1-self.beta1**self.t)
            v_hat = self.v[key] / (1-self.beta2**self.t)

            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    def zero_grad(self):
        pass
class FullyConnectedNet():
    def __init__(self,input_size,hidden_size,output_size):
        self.params = {}
        self.params['W1'] = np.random.randn(input_size,hidden_size) * 0.01
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size,output_size) * 0.01
        self.params['b2'] = np.zeros(output_size)

    def relu(self,x):
        return np.maximum(0,x)
    def forward(self,X):
        W1,b1 = self.params['W1'],self.params['b1']
        W2,b2 = self.params['W2'],self.params['b2']

        z1 = X @ W1 + b1
        a1 = self.relu(z1)

        scores = a1 @ W2 + b2

        self.cache =(X,z1,a1,scores)
        return scores

    def compute_loss(self, scores, y_true):
        N = scores.shape[0]
        loss = 0.5 * np.mean((scores - y_true) ** 2)
        return loss

    def backward(self, y_true):
        X, z1, a1, scores = self.cache
        N = X.shape[0]
        W1,b1 = self.params['W1'], self.params['b1']
        W2,b2 = self.params['W2'], self.params['b2']
        grads = {}

        dL_dy = (scores - y_true) /N

        grads['W2'] = a1.T @ dL_dy
        grads['b2'] = np.sum(dL_dy, axis=0)

        da1 = dL_dy @ W2.T
        dz1 = da1 * (z1 > 0)

        grads['W1'] = X.T @ dz1
        grads['b1'] = np.sum(dz1, axis=0)
        return grads
    def numerical_gradient(self, X, y_true,eps=1e-6):
        grad_nums = {}
        for param_name in self.params:
            param =self.params[param_name]
            grad_num = np.zeros_like(param)
            it = np.nditer(grad_num, flags=['multi_index'],op_flags=['readwrite'])
            while not  it.finished:
                idx = it.multi_index

                original = param[idx]
                param[idx] = original + eps
                scores_high = self.forward(X)
                loss_high = self.compute_loss(scores_high, y_true)

                param[idx] = original- eps
                scores_low = self.forward(X)
                loss_low = self.compute_loss(scores_low, y_true)

                grad_num[idx] = (loss_high-loss_low)/(2*eps)

                param[idx] = original
                it.iternext()
            grad_nums[param_name] = grad_num
        return grad_nums


def rel_error(x, y):
    return np.max(np.abs(x-y)/(np.maximum(1e-8,np.abs(x)+np.abs(y))))

if __name__ == '__main__':
    input_size = 10
    hidden_size = 20
    output_size = 1
    net = FullyConnectedNet(input_size,hidden_size,output_size)
    optimizer = AdamOptimizer(net.params,lr=1e-3)


    X = np.random.randn(5,input_size)
    y_true = np.random.randn(5,output_size)

    scores = net.forward(X)
    loss = net.compute_loss(scores, y_true)
    print(f"更新前loss:{loss:.6f}")

    grads = net.backward(y_true)
    optimizer.step(grads)

    scores = net.forward(X)
    loss = net.compute_loss(scores, y_true)
    print(f"更新后loss:{loss:.6f}")
