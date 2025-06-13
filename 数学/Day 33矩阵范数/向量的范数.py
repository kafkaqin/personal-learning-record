import numpy as np

v = np.array([3,4])
l1 = np.linalg.norm(v,1)
print(l1)
l2 = np.linalg.norm(v,2)
print(l2)
linf = np.linalg.norm(v,np.inf)
print(linf)