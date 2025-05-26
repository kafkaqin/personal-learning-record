
在统计学中，**t检验（t-test）** 是一种常用的假设检验方法，用于比较两个样本组的均值是否存在显著差异。

Python 的 **SciPy 库** 提供了多种 t 检验函数，其中最常用的是：

- `scipy.stats.ttest_ind()`：独立样本 t 检验（Independent t-test）
- `scipy.stats.ttest_rel()`：配对样本 t 检验（Paired t-test）
- `scipy.stats.ttest_1samp()`：单样本 t 检验（One-sample t-test）

---

## ✅ 一、独立样本 t 检验示例

适用于两组来自不同群体的数据，判断它们的均值是否显著不同。

### 🧪 示例代码：

```python
from scipy.stats import ttest_ind
import numpy as np

# 假设我们有两个班级的学生数学成绩
group_a = np.random.normal(loc=75, scale=10, size=30)  # 班级A，均值75，标准差10
group_b = np.random.normal(loc=82, scale=10, size=30)  # 班级B，均值82，标准差10

# 进行独立样本 t 检验
t_stat, p_value = ttest_ind(group_a, group_b)

print("t 统计量:", t_stat)
print("p 值:", p_value)

if p_value < 0.05:
    print("结论：两组数据均值存在显著差异（p < 0.05）")
else:
    print("结论：两组数据均值无显著差异（p ≥ 0.05）")
```

---

## ✅ 二、配对样本 t 检验示例

适用于同一组对象在两种条件下测量得到的数据（如治疗前后）。

### 🧪 示例代码：

```python
from scipy.stats import ttest_rel

# 假设记录某药物治疗前后病人的血压值
before = np.array([140, 150, 145, 155, 160, 158, 149, 152])
after = np.array([135, 145, 140, 150, 155, 152, 145, 148])

# 配对样本 t 检验
t_stat, p_value = ttest_rel(before, after)

print("t 统计量:", t_stat)
print("p 值:", p_value)

if p_value < 0.05:
    print("结论：治疗前后血压存在显著差异（p < 0.05）")
else:
    print("结论：治疗前后血压无显著差异（p ≥ 0.05）")
```

---

## ✅ 三、单样本 t 检验示例

用于判断一个样本的均值是否与某个已知总体均值有显著差异。

### 🧪 示例代码：

```python
from scipy.stats import ttest_1samp

# 假设我们知道全国平均 IQ 为 100
sample_iq = np.random.normal(loc=105, scale=15, size=40)  # 测量40人IQ

# 单样本 t 检验
t_stat, p_value = ttest_1samp(sample_iq, popmean=100)

print("t 统计量:", t_stat)
print("p 值:", p_value)

if p_value < 0.05:
    print("结论：样本均值与总体均值存在显著差异（p < 0.05）")
else:
    print("结论：样本均值与总体均值无显著差异（p ≥ 0.05）")
```

---

## 📌 四、结果解读说明

| 名称 | 含义 |
|------|------|
| `t 统计量` | 表示两组之间的差异程度，绝对值越大差异越明显 |
| `p 值` | 衡量统计显著性的指标。通常以 0.05 为显著性阈值 |
| `p < 0.05` | 拒绝原假设，认为两组之间存在显著差异 |
| `p ≥ 0.05` | 无法拒绝原假设，即没有足够证据显示差异 |

---

## ⚠️ 注意事项

- 样本应尽量满足正态分布，尤其在小样本情况下。
- 独立样本 t 检验默认方差齐性（homogeneity of variance），若不满足可使用 `equal_var=False` 参数。
- 若数据不服从正态分布，考虑使用非参数检验，如 Mann-Whitney U 检验或 Wilcoxon 符号秩检验。

---

## 📊 扩展推荐

- 可视化对比：箱线图（boxplot）、误差条图（errorbar）
- 效应大小（effect size）计算，如 Cohen's d
- 多组比较使用 ANOVA（方差分析）

---

如果你有实际的数据文件（CSV/Excel）或者具体问题需要进行 t 检验，我可以帮你写出完整的分析流程和可视化代码 😄  
欢迎继续提问！