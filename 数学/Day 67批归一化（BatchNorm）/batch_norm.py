import torch
import torch.nn as nn
import torch.nn.functional as F

class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MyBatchNorm1d, self).__init__()
        self.eps = eps
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))\

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0,unbiased=False)

            self.running_mean =  (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            self.num_batches_tracked = self.num_batches_tracked + 1
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        out = self.gamma * x_norm + self.beta
        return out


class SimpleModelWithBN(nn.Module):
    def __init__(self):
        super(SimpleModelWithBN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 50),
            MyBatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, x):
        return self.net(x)


class SimpleModelWithoutBN(nn.Module):
    def __init__(self):
        super(SimpleModelWithoutBN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
    def forward(self, x):
        return self.net(x)

import torch.optim as optim

def get_data(num_samples=1000):
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    return X, y

X , y = get_data()

model_bn = SimpleModelWithBN()
model_no_bn = SimpleModelWithoutBN()

criterion = nn.MSELoss()
optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.01)
optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=0.01)


def train(model,optimizer,X,y,epochs=200,model_name="Model"):
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch+1) % 50 == 0:
            print(f"{model_name} Epoch [{epoch+1}/{epochs}], Loss:{loss.item():.4f}")

        return losses


import matplotlib.pyplot as plt

losses_bn = train(model_bn,optimizer_bn,X,y,model_name="Model_BN")
losses_no_bn = train(model_no_bn,optimizer_no_bn,X,y,model_name="Model No_BN")

plt.plot(losses_bn, label="Without BatchNorm")
plt.plot(losses_no_bn, label="Without BatchNorm")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title('Training Loss with/without BatchNorm')
plt.grid(True)
plt.savefig('Loss.png')