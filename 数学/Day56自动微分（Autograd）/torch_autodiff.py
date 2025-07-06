import torch
x = torch.tensor(0.0, requires_grad=True)
f = x**2 + torch.sin(x)
f.backward()
print("f'(0) =",x.grad)

x = torch.tensor([1.0,2.0], requires_grad=True)
f = x[0]**2 + x[1]**3
f.backward()
print("f'(0) =",x.grad)