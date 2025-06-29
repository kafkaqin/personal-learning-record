要使用Python模拟布朗运动路径，我们可以采用随机微积分的基本概念。在离散时间步骤中，布朗运动可以通过一系列的独立正态分布（高斯分布）随机变量来近似实现。每个时间步长中的位移是一个平均值为0、方差与时间步长成比例的正态分布随机数。

下面我将给出一个简单的Python代码示例，用于生成并绘制一条或多条布朗运动路径：

### Python代码示例

```python
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
T = 1.0           # 总时间
N = 1000          # 时间步数
dt = T / N        # 每一步的时间
t = np.linspace(0, T, N)  # 时间轴

# 初始化位置
position = np.zeros(N)

# 对于每一步，我们添加一个均值为0，标准差为sqrt(dt)的正态分布随机变量
dW = np.random.normal(0, np.sqrt(dt), N)
position = np.cumsum(dW)  # 累积和代表位置随时间的变化

# 绘制单条布朗运动路径
plt.figure(figsize=(10, 6))
plt.plot(t, position, label="布朗运动路径", lw=2)
plt.title('布朗运动路径模拟')
plt.xlabel('时间')
plt.ylabel('位置')
plt.legend()
plt.show()

# 若要模拟多条路径，可以这样做：
num_paths = 5  # 要模拟的路径数量
plt.figure(figsize=(10, 6))

for i in range(num_paths):
    dW = np.random.normal(0, np.sqrt(dt), N)
    position = np.cumsum(dW)
    plt.plot(t, position, lw=1.5, alpha=0.7)

plt.title(f'{num_paths}条布朗运动路径模拟')
plt.xlabel('时间')
plt.ylabel('位置')
plt.show()
```

这段代码首先定义了总时间和步数，并根据这些参数计算出每一步的时间间隔`dt`。然后，它为每一个时间步长生成一个符合正态分布的随机变量，其标准差是`sqrt(dt)`，表示该步长内位置变化的标准偏差。通过累积求和这些随机变量，我们得到一个随时间变化的位置序列，这便是布朗运动的一条路径。最后，使用Matplotlib库绘制这条路径。

若想观察多条路径的行为，只需重复上述过程多次，并在同一个图上绘制所有路径即可。这样可以帮助理解布朗运动的随机性和扩散特性。