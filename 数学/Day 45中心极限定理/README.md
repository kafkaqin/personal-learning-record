中心极限定理（Central Limit Theorem, CLT）是概率论中的一个重要定理，它指出，在一定条件下，大量相互独立的随机变量之和近似服从正态分布。为了通过抽样模拟验证这一理论，我们可以执行以下步骤：

1. **选择一个分布**：这个分布可以是非正态的，比如均匀分布、指数分布等。
2. **从该分布中抽取样本**：每次抽取一定数量的观测值作为一个样本，并计算其均值。
3. **重复上述步骤多次**：生成大量的样本均值。
4. **观察样本均值的分布**：根据中心极限定理，这些样本均值的分布应该接近于正态分布。

下面是使用 Python 来实现这一过程的具体示例：

### 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
sample_size = 30  # 每个样本的大小
num_samples = 1000  # 样本的数量
distribution = 'uniform'  # 可以选择'uniform', 'exponential'

# 根据选定的分布生成数据
if distribution == 'uniform':
    data = [np.mean(np.random.uniform(0, 1, sample_size)) for _ in range(num_samples)]
elif distribution == 'exponential':
    data = [np.mean(np.random.exponential(scale=1.0, size=sample_size)) for _ in range(num_samples)]

# 绘制直方图
plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

# 计算均值和标准差
mu, sigma = np.mean(data), np.std(data)

# 在直方图上绘制相应的正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5*((x - mu)/sigma)**2)
plt.plot(x, p, 'k', linewidth=2)
title = f"Sample Means Distribution (Sample Size={sample_size}, Samples={num_samples})"
plt.title(title)
plt.xlabel('Mean Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()

print(f"Sample Mean: {mu}")
print(f"Standard Deviation of Sample Means: {sigma}")
```

### 输出解释

- **直方图**：展示了所抽取样本均值的分布情况。
- **正态分布曲线**：基于所有样本均值的平均值和标准差绘制，用来对比样本均值的分布是否接近正态分布。
- **输出信息**：包括了样本均值的平均值和标准差，理论上，随着样本量的增加，样本均值的标准差会逐渐减小。

### 注意事项

- **样本大小的选择**：一般来说，当样本大小至少为 30 时，CLT 开始显现效果，但具体大小取决于原始分布的形态。
- **分布类型的影响**：对于某些分布（如非常偏斜或有重尾的分布），可能需要更大的样本大小才能看到正态分布的趋势。
- **实验重复性**：由于随机性的作用，每次运行代码得到的结果可能会有所不同，但这不影响整体趋势。

通过这种方式，你可以直观地看到无论原始分布如何，只要样本足够大，样本均值的分布都会趋向于正态分布，这正是中心极限定理的核心内容。如果你希望尝试不同的分布或其他参数设置