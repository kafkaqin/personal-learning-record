当然可以！下面是一个使用 `scikit-optimize`（简称 `skopt`）对机器学习模型的超参数进行优化的完整示例。

我们将使用 **随机森林分类器**（Random Forest）在 `Iris` 数据集上，通过 `skopt` 的贝叶斯优化（Bayesian Optimization）来寻找最优超参数组合。

---

## ✅ 目标

使用 `scikit-optimize` 优化以下超参数：

- `n_estimators`: 决策树数量（整数）
- `max_depth`: 树的最大深度（整数或 None）
- `min_samples_split`: 分裂内部节点所需的最小样本数（整数）
- `min_samples_leaf`: 叶节点所需的最小样本数（整数）

---

## 🧰 安装依赖

```bash
pip install scikit-optimize scikit-learn matplotlib
```

---

## 📊 示例代码：使用 `skopt` 优化随机森林超参数

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 定义搜索空间
dim_n_estimators = Integer(10, 200, name='n_estimators')
dim_max_depth = Integer(1, 20, name='max_depth')
dim_min_samples_split = Integer(2, 20, name='min_samples_split')
dim_min_samples_leaf = Integer(1, 20, name='min_samples_leaf')

dimensions = [dim_n_estimators, dim_max_depth, dim_min_samples_split, dim_min_samples_leaf]

# 默认参数（初始点）
default_params = [100, 10, 2, 1]

# 3. 定义目标函数（要最小化的目标：负的交叉验证准确率）
rf = RandomForestClassifier(random_state=42)

@use_named_args(dimensions)
def objective(**params):
    rf.set_params(**params)
    return -cross_val_score(rf, X, y, cv=5, n_jobs=-1, scoring='accuracy').mean()

# 4. 执行贝叶斯优化
search_result = gp_minimize(
    func=objective,
    dimensions=dimensions,
    acq_func='EI',           # 采集函数：期望改进（Expected Improvement）
    n_calls=50,              # 迭代次数
    x0=default_params,       # 初始参数
    random_state=42
)

# 5. 输出结果
print("最优超参数：")
print(f"n_estimators = {search_result.x[0]}")
print(f"max_depth = {search_result.x[1]}")
print(f"min_samples_split = {search_result.x[2]}")
print(f"min_samples_leaf = {search_result.x[3]}")

print(f"\n最优交叉验证准确率: {-search_result.fun:.4f}")

# 6. （可选）绘制优化过程
from skopt.plots import plot_convergence
plot_convergence(search_result)
plt.show()
```

---

## 📈 输出示例

```
最优超参数：
n_estimators = 148
max_depth = 12
min_samples_split = 5
min_samples_leaf = 2

最优交叉验证准确率: 0.9667
```

`plot_convergence()` 会显示目标函数值随迭代次数下降的趋势，帮助你判断优化是否收敛。

---

## 🔍 说明

| 组件 | 说明 |
|------|------|
| `gp_minimize` | 使用高斯过程（Gaussian Process）进行贝叶斯优化 |
| `Integer` | 定义整数型超参数搜索空间 |
| `@use_named_args` | 允许使用命名参数传递给目标函数 |
| `acq_func='EI'` | 使用“期望改进”策略选择下一个采样点 |
| `n_calls` | 控制优化迭代次数，越多越可能找到最优，但耗时更长 |

---

## ✅ 优势对比网格搜索/随机搜索

- **更高效**：贝叶斯优化利用历史评估结果建模，智能选择下一个候选点。
- **适合昂贵的评估**：如深度学习模型训练、大规模数据训练等。
- **支持连续、离散、条件空间**。

---

## 🚀 扩展建议

- 结合 `Pipeline` 和 `sklearn` 模型进行端到端优化。
- 使用 `dump` 和 `load` 保存/恢复优化结果。
- 支持条件超参数（例如：仅当 `criterion='tree'` 时才优化 `max_leaf_nodes`）。
- 替换为 `forest_minimize`（随机森林回归器建模）或 `gbrt_minimize`（梯度提升树）。

---