import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = tanh(x)
    return 1-t**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x >=0,x,alpha*x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x >=0,1.0,alpha*x)

def elu(x, alpha=1.0):
    return np.where(x >=0,x,alpha*(np.exp(x)-1))

def elu_derivative(x, alpha=1.0):
    return np.where(x >=0,1.0,alpha*(np.exp(x)))

x = np.linspace(-5,5,400)

activations = [
    ("Sigmoid", sigmoid,sigmoid_derivative),
    ("Tanh", tanh,tanh_derivative),
    ("ReLU", relu,relu_derivative),
    ("Leaky ReLU",lambda x: leaky_relu(x),lambda x:leaky_relu_derivative(x)),
    ("ELU", lambda x:elu(x),lambda x:elu_derivative(x)),
]

plt.figure(figsize=(15, 10))

for i, (name, act_fn, grad_fn) in enumerate(activations):
    y = act_fn(x)
    dy = grad_fn(x)

    # 绘制激活函数
    plt.subplot(len(activations), 2, 2*i+1)
    plt.plot(x, y, label=name)
    plt.title(f"{name} Activation")
    plt.grid(True)
    plt.xlabel("Input")
    plt.ylabel("Output")

    # 绘制梯度
    plt.subplot(len(activations), 2, 2*i+2)
    plt.plot(x, dy, color='orange', label=f"{name} Gradient")
    plt.title(f"{name} Gradient")
    plt.grid(True)
    plt.xlabel("Input")
    plt.ylabel("Derivative")

plt.tight_layout()
plt.savefig("activations.png")