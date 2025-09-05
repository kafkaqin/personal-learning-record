import numpy as np
A = np.array([[1,2],[3,4]])

fro_norm = np.linalg.norm(A)
print(fro_norm)

import numpy as np
v = np.array([3,4])
A = np.array([
    [1,2],
    [3,4],
],dtype=float)
print(v)
print(A)
l1 = np.linalg.norm(v, ord=1)
l2 = np.linalg.norm(v,ord=2)
print(l1,l2)
linf = np.linalg.norm(v,ord=np.inf)
print(linf)
l3 = np.linalg.norm(v,ord=3)
print(l3)

fro = np.linalg.norm(A,ord='fro')
print(fro)
spec = np.linalg.norm(A,ord=2)
print(spec)
col_norm = np.linalg.norm(A,ord=1)
print(col_norm)
row_norm = np.linalg.norm(A,ord=2)
print(row_norm)
nuc = np.linalg.norm(A,ord='nuc')
print(nuc)

fro_manual = np.sqrt(np.sum(A**2))
print(fro_manual)
print(f"{np.allclose(fro,fro_manual)}")

W = np.array([0.1,-0.5,2.0,0.01])
l1_loss = np.linalg.norm(W,ord=1)
print(l1_loss)
l2_loss = np.linalg.norm(W,ord=2)**2
print(l2_loss)

img1 = np.array([[100,105],[110,115]])
img2 = np.array([[102,104],[108,118]])
diff = img1-img2
error = np.linalg.norm(diff,'fro')
print(error)
U,sigma,Vt = np.linalg.svd(A)
k = 1
A_approx = U[:,:k] @ np.diag(sigma[:k]) @ Vt[:k,:]
approx_error = np.linalg.norm(A-A_approx,'fro')
print(approx_error)