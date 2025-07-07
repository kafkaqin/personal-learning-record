使用 **JAX** 或 **PyTorch** 实现自动求导（Automatic Differentiation）是现代机器学习和科学计算中的核心能力。两者都支持高效的梯度计算，但它们的实现方式和适用场景略有不同。

---

## 🧠 一、自动求导简介

自动求导（AutoDiff）是一种通过跟踪或编译函数执行路径来高效计算导数的技术。它比数值微分更精确，也比符号微分更适合大规模计算。

- **前向模式（Forward Mode）**：适合输入维度小的问题。
- **反向模式（Reverse Mode）**：适合神经网络等输出为标量、输入维度大的问题。

---

## ✅ 二、用 JAX 实现自动求导

JAX 是一个用于高性能数值计算的库，内置了基于 **XLA 编译器** 的自动求导系统，并支持 JIT 加速、向量化等特性。

### 示例：对函数 $ f(x) = x^2 + \sin(x) $ 求导

```python
import jax
import jax.numpy as jnp

# 定义函数
def f(x):
    return x**2 + jnp.sin(x)

# 使用 grad 自动求导
df = jax.grad(f)

# 计算在 x=0 处的导数
x = 0.0
print("f'(0) =", df(x))  # 应该是 0 + cos(0) = 1
```

### 更高级用法：

```python
# 二阶导数
d2f = jax.grad(jax.grad(f))
print("f''(0) =", d2f(x))  # -sin(0) = 0

# 向量化版本（批量输入）
vmap_df = jax.vmap(df)
xs = jnp.array([0.0, jnp.pi/2, jnp.pi])
print("批量导数:", vmap_df(xs))
```

---

## ✅ 三、用 PyTorch 实现自动求导

PyTorch 是一个专注于深度学习的框架，其 `torch.autograd` 模块提供了强大的自动求导功能，适用于构建动态计算图。

### 示例：对函数 $ f(x) = x^2 + \sin(x) $ 求导

```python
import torch

# 创建张量并启用自动求导
x = torch.tensor(0.0, requires_grad=True)

# 定义函数
f = x**2 + torch.sin(x)

# 反向传播求导
f.backward()

# 获取导数
print("f'(0) =", x.grad)  # 应该是 0 + cos(0) = 1
```

### 对于多变量函数：

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
f = x[0]**2 + x[1]**3

f.backward()
print("梯度:", x.grad)  # [2*1, 3*(2)^2] = [2., 12.]
```

---

## 📌 四、JAX vs PyTorch 自动求导对比

| 特性 | JAX | PyTorch |
|------|-----|----------|
| 核心定位 | 数值计算、科学计算 | 深度学习、AI |
| 是否需要声明 `requires_grad` | ❌ 不需要 | ✅ 需要 |
| 支持 JIT 编译加速 | ✅ 强大支持 | ❌（需 TorchScript 或 TorchDynamo） |
| 函数式编程风格 | ✅ 主流风格 | ❌ 更偏向面向对象 |
| GPU/TPU 支持 | ✅ 原生支持 | ✅ 强大支持 |
| 动态图/静态图 | ✅ 全动态 | ✅ 默认动态（可转换为静态） |
| 梯度计算方式 | `grad(func)` | `.backward()` / `torch.autograd.grad` |

---

## 🧪 五、应用场景建议

| 场景 | 推荐工具 |
|------|----------|
| 神经网络训练 | PyTorch |
| 科学模拟、物理建模 | JAX |
| 微分方程求解 | JAX（如 DiffEqFlux） |
| 快速原型开发 | PyTorch（生态丰富） |
| 高性能数值计算 | JAX（结合 JIT 和 XLA） |

---

## ✅ 六、扩展：使用 `jax.value_and_grad` 或 `torch.func.grad`

### JAX：同时获取值和梯度

```python
from jax import value_and_grad

val, grad_val = value_and_grad(f)(0.0)
print("f(0) =", val)
print("f'(0) =", grad_val)
```

### PyTorch：使用 `torch.func.grad`（v2.0+）

```python
from torch.func import grad

f = lambda x: x**2 + torch.sin(x)
df = grad(f)
print("f'(0) =", df(torch.tensor(0.0)))
```

---