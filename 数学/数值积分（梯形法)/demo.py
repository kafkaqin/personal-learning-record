def trapezoidal_rule(f, a, b, n):
    """
    梯形法则计算定积分 ∫_a^b f(x) dx

    参数:
    f : 被积函数 (Python 函数)
    a : 积分下限
    b : 积分上限
    n : 子区间数量（分割点数为 n + 1）

    返回:
    近似积分值
    """
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))  # 第一项和最后一项各乘 0.5
    for i in range(1, n):
        x = a + i * h
        total += f(x)
    return total * h


# 测试函数：f(x) = x^2
def f(x):
    return x ** 2


# 设置参数
a = 0  # 积分下限
b = 1  # 积分上限
n = 100  # 分割的子区间数量

# 计算积分
integral = trapezoidal_rule(f, a, b, n)

# 输出结果
print("近似积分值:", integral)
print("误差:", abs(integral - 1 / 3))