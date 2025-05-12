### Symbolic Differentiation
import sympy as sp

if __name__ == '__main__':
    x,y = sp.symbols('x y')
    g = x**2*y + sp.sin(y)
    dg_dx = sp.diff(g,x,1)
    print(dg_dx)

    dg_dy = sp.diff(g,y,2)
    print(dg_dy)
