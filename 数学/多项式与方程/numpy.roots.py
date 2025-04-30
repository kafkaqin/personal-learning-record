import numpy as np
    #
    a = 1
    b = 5
    c = 6
    roots = np.roots([a,b,c])
    print(roots)


# import numpy as np
# import matplotlib.pyplot as plt
# from sympy import symbols,solve
# if __name__=='__main__':
#
#
#     x = np.linspace(-30,30,500)
#
#     y_sin = np.sin(x)
#     y_exp = np.exp(x)
#     y_cos = np.cos(x)
#     y_tan = np.tan(x)
#     plt.figure(figsize=(12,6))
#
#     plt.subplot(2,3,1)
#     plt.plot(x,y_sin,label='sin',color='blue')
#     plt.title("Sine Function")
#     plt.xlabel("x")
#     plt.ylabel("sin(x)")
#     plt.legend()
#
#     plt.subplot(2,3,2)
#     plt.plot(x,y_exp,label='exp',color='red')
#     plt.title("Exponent Function")
#     plt.xlabel("x")
#     plt.ylabel("exp(x)")
#     plt.legend()
#
#
#     plt.subplot(2,3,3)
#     plt.plot(x,y_cos,label='cos',color='green')
#     plt.title("Cosine Function")
#     plt.xlabel("x")
#     plt.ylabel("cos(x)")
#     plt.legend()
#
#
#     plt.subplot(2,3,4)
#     plt.plot(x,y_tan,label='tan',color='blue')
#     plt.title("Tan Function")
#     plt.xlabel("x")
#     plt.ylabel("tan(x)")
#
#     plt.tight_layout()
#     # plt.show()
#     plt.savefig('test.png')
#
#     #
#     a = 1
#     b = -4
#     c = 1
#     d = 9
#     coeffs = [a,b,c,d,9]
#     roots = np.roots(coeffs)
#     print(roots)
#
#     root = roots[0]
#     value = np.polyval(coeffs,root)
#     print(f"poly{root} = {value}")
#
#     x = symbols('x')
#     poly_expr = x**4 - 3*x**2+1
#     roots_value = solve(poly_expr,x)
#     print(f"roots_value = {roots_value}")