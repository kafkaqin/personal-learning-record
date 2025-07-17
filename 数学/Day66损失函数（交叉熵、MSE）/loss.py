import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(42)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1,bias=True)
    def forward(self, x):
        return self.linear(x)
model = SimpleModel()
print("初始模型参数: ")
print(model.state_dict())

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
],requires_grad=True)

targets = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
])

outputs = model(inputs)
loss = criterion(outputs, targets)

print("\n前向转播输出:")
print(outputs)
print("\n计算损失:",loss.item())
optimizer.zero_grad()
loss.backward()

print("\n参数梯度:")
for name,param in model.named_parameters():
    print(f"{name} : {param.grad}")