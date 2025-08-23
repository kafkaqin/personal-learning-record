import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(0,10,50)
y = 3 * X +7+ np.random.normal(0,2,size=X.shape)
X_design = np.column_stack([np.ones(X.shape),X])

beta,residuals,rank,s=np.linalg.lstsq(X_design,y,rcond=None)
beta_0,beta_1 = beta
print(f"拟合的回归方程: y={beta_1:.2f}x+{beta_0:.2f}")

y_pred = X_design @ beta

ss_res = np.sum((y-y_pred) **2)
ss_tot = np.sum((y-np.mean(y))**2)
r_squared = 1 - (ss_res/ss_tot)
print(f"R2: {r_squared:.4f}")


np.random.seed(123)
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = np.random.randn(n_samples)
y_multi = 2 * X1 + 3*X2 +5+np.random.randn(n_samples) * 0.5

X_design_multi = np.column_stack([np.ones(n_samples),X1,X2])
beta_multi,_,_,_ = np.linalg.lstsq(X_design_multi,y_multi,rcond=None)
intercent,coef1,coef2 = beta_multi
print(f"截距:{intercent}\n")
print(f"X1的系数{coef1:.2f}")
print(f"X2的系数{coef2:.2f}")

XTX = X_design.T @ X_design
XTy = X_design.T @ y
beta_manual = np.linalg.inv(XTX) @ XTy
print(beta_manual)