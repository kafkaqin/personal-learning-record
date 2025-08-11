import sympy as sp

x = sp.symbols('x')

f = x**2
df_dx = sp.diff(f,x)
print(df_dx)

d2f_dx2 = sp.diff(f,x,2)
print(d2f_dx2)

# sin(x), exp(x), ln(x) 等
f = sp.sin(x) * sp.exp(x)
df_dx = sp.diff(f, x)
print("d/dx(sin(x)*exp(x)) =", df_dx)

x, y = sp.symbols('x y')
f = x**2 * y + sp.sin(x)

# 对 x 求偏导
df_dx = sp.diff(f, x)
print("∂f/∂x =", df_dx)

# 对 y 求偏导
df_dy = sp.diff(f, y)
print("∂f/∂y =", df_dy)

t = sp.symbols('t')
x = sp.sin(t)
f = x**2  # f = sin^2(t)

# 自动应用链式法则
df_dt = sp.diff(f, t)
print("d/dt(sin^2(t)) =", df_dt)

x, y = sp.symbols('x y')
y = sp.Function('y')(x)  # y 是 x 的函数

eq = x**2 + y**2 - 1

# 对两边求导
d_eq = sp.diff(eq, x)
# 解出 dy/dx
dy_dx = sp.solve(d_eq, sp.Derivative(y, x))[0]
print("dy/dx =", dy_dx)