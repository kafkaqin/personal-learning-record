import numpy as np
import matplotlib.pyplot as plt

# 时间域转换到频率域
if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("Matrix A", A)
    print("Matrix B", B)

    add_result = A + B
    print("A+B=", add_result)

    sub_result = A - B
    print("A - B =", sub_result)
    matmul_result = np.matmul(A, B)
    print("AmatmulB=", matmul_result)
    print("A@B=", A @ B)
    print("A*B=", A * B)
