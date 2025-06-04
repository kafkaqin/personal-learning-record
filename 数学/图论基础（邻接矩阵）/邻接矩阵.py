import numpy as np

graph = np.array([
    [np.inf,1,4,np.inf],
    [np.inf,np.inf,2,6],
    [np.inf,3,np.inf,1],
    [np.inf,np.inf,np.inf,np.inf],
])
print("邻接矩阵:")
print(graph)