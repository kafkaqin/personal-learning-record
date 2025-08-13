import sympy as sp

x  = sp.symbols('x')

taylor_sin = sp.series(sp.sin(x),x,0,5)
print(taylor_sin)

taylor_cos = sp.series(sp.cos(x),x,0,6)
print(taylor_cos)

taylor_exp = sp.series(sp.exp(x),x,0,4)
print(taylor_exp)

taylor_log =sp.series(sp.log(x),x,0,5)
print(taylor_log)

a = sp.pi / 2
taylor_sin_at_pi2 = sp.series(sp.sin(x),x,a,4)
print(taylor_sin_at_pi2)