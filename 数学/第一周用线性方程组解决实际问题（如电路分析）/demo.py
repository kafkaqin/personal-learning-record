import numpy as np

A = np.array([
    [1,1,-1],
    [2,0,1],
    [0,4,1]
])

b = np.array([0,10,6])

try:
    x = np.linalg.solve(A,b)
    I1,I2,I3 = x
    print(I1,I2,I3)
except np.linalg.LinAlgError:
    print('LinAlgError')