import torch
from IPython import display
from d2l import torch as d2l
# batch_size = 256
# train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
#
num_input = 784
num_output = 10

W = torch.normal(0, 0.01, (num_input, num_output), requires_grad=True)
b = torch.zeros(num_output, requires_grad=True)

X = torch.tensor([[1,2,3],[4,5,6]])
print(X.sum(0,keepdim=True))
print(X.sum(1,keepdim=True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp/partition

X = torch.normal(0,0.01,(2,5))
X_prob = softmax(X)
print(X_prob,X_prob.sum(1, keepdim=True))

def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y_hat[[0,1],y]

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)),y])

print(cross_entropy(y_hat,y))

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

print(accuracy(y_hat,y))

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

