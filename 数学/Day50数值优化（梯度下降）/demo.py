import numpy as np
np.random.seed(0)
X= 2 * np.random.randn(100,1)
y = 4+3*X+np.random.randn(100,1)

m = 0
b = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

for epoch in range(epochs):
    gradient_m = -(2/n)*np.sum(X*(y-(m*X+b)))
    gradient_b = -(2/n)*np.sum(y-(m*X+b))

    m -=learning_rate *gradient_m
    b -=learning_rate* gradient_b
    if epoch % 100 == 0:
        loss = np.sum((y-(m*X+b))**2)/n
        print('Epoch:',epoch,'Loss:',loss)

print("最终参数: 斜率 =",m,"截距 =",b)

