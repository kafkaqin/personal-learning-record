import numpy as np
def softmax(z):
    exp_z = np.exp(z-np.max(z,axis=1,keepdims=True))
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)


def cross_entropy_loss(y_true,y_pred):
    m = y_true.shape[0]
    log_likelihood  = -np.log(y_pred[range(m),y_true.argmax(axis=1)]+1e-15)
    loss = np.mean(log_likelihood)
    return loss
np.random.seed(0)
X = np.random.randn(6,4)

y_true = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [0,1,0],
    [0,0,1],
])

W = np.random.randn(4,3) * 0.01
b = np.zeros((1,3))

logits = X @ W + b

y_pred = softmax(logits)


loss = cross_entropy_loss(y_true,y_pred)
print("交叉商损失: ",loss)


learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    logits = X @ W + b
    y_pred = softmax(logits)
    loss = cross_entropy_loss(y_true,y_pred)

    if epoch % 100 == 0:
        print("epoch:",epoch,"loss:",loss,"lr:",learning_rate)

    grad_logits = y_pred-y_true
    grad_W = X.T @ grad_logits
    grad_b = np.sum(grad_logits,axis=0,keepdims=True)

    W -= learning_rate * grad_W
    b-= learning_rate * grad_b