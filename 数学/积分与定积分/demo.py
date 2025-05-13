import sympy as sp

# 定义变量
x = sp.symbols('x')

# 定义被积函数
f = x**2 + 2*x + 1

# 计算定积分
integral_result = sp.integrate(f, (x, 0, 1))

print(integral_result)