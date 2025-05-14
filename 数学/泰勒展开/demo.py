import sympy as sp

x = sp.symbols('x')
f = sp.sin(x)

# 在 x = 0 处展开，到 x^5 为止
taylor_series = sp.series(f, x, 0, 5)
print(taylor_series)

# 在 x = π/2 处展开 cos(x)，到 x^4 阶
taylor_series = sp.series(sp.cos(x), x, sp.pi/2, 4)
print(taylor_series)

poly_part = sp.series(sp.exp(x), x, 0, 4).removeO()
print(poly_part)