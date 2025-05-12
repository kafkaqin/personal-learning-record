### Symbolic Differentiation
import sympy as sp

if __name__ == '__main__':
    x = sp.symbols('x')
    h = sp.sin(x**2)
    # 求导
    dh = sp.diff(h, x)
    print(dh)  #