import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import CrossEntropyLoss

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hidden = 784,10,256

W1 = nn.Parameter(torch.randn(num_inputs,num_hidden,requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hidden,requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hidden,num_outputs,requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
params = [W1,b1,W2,b2]

def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)

def net(X):
    X = X.reshape((-1,num_outputs))
    H1 = W1@X+b1
    H1 = relu(H1)
    H2 = W2@H1+b2
    return H2

loss = CrossEntropyLoss()
num_epochs,lr = 10,0.1
updater = torch.optim.SGD(params,lr=lr)
d2l.train_ch6(net, train_iter, test_iter,  num_epochs, loss,updater)


net = nn.Sequential(nn.Flatten(),nn.Linear(num_inputs,num_hidden),nn.ReLU(),nn.Linear(num_hidden,num_outputs))
loss = CrossEntropyLoss()
num_epochs,lr = 10,0.1
updater = torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, loss, updater)
