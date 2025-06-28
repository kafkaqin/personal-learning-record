使用 **置信区间（Confidence Interval, CI）** 是统计推断中非常常见的一种方法，用于估计总体参数（如均值）的可能范围。在 Python 中，可以使用 `scipy.stats.norm.interval` 来基于正态分布计算置信区间。

---

## ✅ 一、什么是置信区间？

一个 **95% 置信区间** 的含义是：

> 如果我们从同一总体中反复抽样并计算置信区间，大约 95% 的置信区间会包含真实总体均值。

---

## 🧪 二、Python 示例：用 `norm.interval` 计算置信区间

### 🔧 使用条件：

- 假设样本来自正态分布或样本量足够大（中心极限定理）
- 已知样本均值和标准误（standard error）

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 模拟数据：从正态分布中生成样本
np.random.seed(42)
sample = np.random.normal(loc=100, scale=15, size=100)  # 总体均值=100，标准差=15，样本量=100

# 样本统计量
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)  # 样本标准差（无偏估计）
n = len(sample)

# 标准误（Standard Error）
se = sample_std / np.sqrt(n)

# 设置置信水平（例如 95%）
confidence_level = 0.95

# 使用 norm.interval 计算置信区间
ci = norm.interval(confidence_level, loc=sample_mean, scale=se)

print(f"样本均值: {sample_mean:.2f}")
print(f"{int(confidence_level * 100)}% 置信区间: {ci}")
```

### ✅ 输出示例：

```
样本均值: 101.89
95% 置信区间: (98.967, 104.813)
```

---

## 📊 三、可视化置信区间

我们可以绘制一个图来展示样本均值及其置信区间：

```python
plt.figure(figsize=(8, 4))
plt.errorbar(
    x=0,
    y=sample_mean,
    yerr=(ci[1] - ci[0]) / 2,
    fmt='o',
    ecolor='r',
    capsize=10,
    color='blue',
    label='Sample Mean with 95% CI'
)

plt.xlim(-1, 1)
plt.ylim(sample_mean - 10, sample_mean + 10)
plt.axhline(y=100, color='gray', linestyle='--', label='True Population Mean')
plt.xticks([])
plt.ylabel("Value")
plt.title("95% Confidence Interval")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🧠 四、扩展说明

| 方法 | 描述 |
|------|------|
| `norm.interval(alpha, loc=mu, scale=se)` | 正态分布下的置信区间（适合大样本或已知总体标准差） |
| `t.interval(alpha, df=n-1, loc=mu, scale=se)` | t 分布下的置信区间（适合小样本） |

### ✅ 小样本建议改用 t 分布：

```python
from scipy.stats import t

ci_t = t.interval(confidence_level, df=n-1, loc=sample_mean, scale=se)
```

---

## 📌 五、应用场景举例

| 场景 | 应用方式 |
|------|----------|
| A/B 测试 | 比较两个组的均值差异是否显著 |
| 调查分析 | 给出调查结果的误差范围 |
| 医学研究 | 估计药物效果的置信范围 |
| 质量控制 | 判断产品性能是否在可接受范围内 |

---