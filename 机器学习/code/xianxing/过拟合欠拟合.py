import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

max_degree = 20
n_train,n_test = 100,100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

features = np.random.normal(size=(n_train+n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,1] /= math.gamma(i+1)
labels = np.dot(poly_features,true_w)
labels+=np.random.normal(0,scale=0.1,size=labels.shape)

true_w,features,poly_features,labels =[torch.tensor(x,dtype=torch.float32) for x in [true_w,features,poly_features,labels]]
features[:2]
poly_features[:2,:]
labels[:2]
def evaluate_loss(net, train_iter, loss):
    metric = d2l.Accumulator(2)
    for X, y in train_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.mumel())
    return metric[0] /metric[1]

def train(train_features, test_features, train_labels, test_labels,num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array(train_features,train_labels.reshape(-1,1),batch_size)
    test_iter = d2l.load_array(test_features,test_labels.reshape(-1,1),batch_size,is_train=False)
    trainer = torch.optim.SGD(net.parameters(),lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', ylim=[0,1])

    for epoch in range(num_epochs):
        d2l.train_ch6(net, train_iter, test_iter, loss, trainer, epoch) 