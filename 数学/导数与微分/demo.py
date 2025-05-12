### Symbolic Differentiation
import sympy as sp

if __name__ == '__main__':
    x = sp.symbols('x')
    f = x**2 + x + 1
#     f = x
    print(f)
    df = sp.diff(f,x,1)
    print(df)

    d2f = sp.diff(f,x,2)
    print(d2f)
