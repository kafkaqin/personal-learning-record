import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
true_w = torch.tensor([2,-3.4])
true_b = 4.2
num_examples = 1000
features,labels = d2l.synthetic_data(true_w,true_b,num_examples)
def load_array(data_arrays,batch_size,is_train=True):
    data_set = data.TensorDataset(*data_arrays)
    return data.DataLoader(data_set,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size,is_train=True)
next(iter(data_iter))

from torch import nn

net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
SGD = torch.optim.SGD(net.parameters(),lr=0.10)

num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        SGD.zero_grad()
        l.backward()
        SGD.step()
    l = loss(net(features),labels)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {l.item():.4f}')