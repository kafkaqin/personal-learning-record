import numpy as np

if __name__ == '__main__':
    A = np.array([[1, 2], [4, 5]])

    b = np.array([10, 11])

    x = np.linalg.solve(A, b)
    print(x)