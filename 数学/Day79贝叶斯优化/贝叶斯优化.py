import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

iris = load_iris()
X, y = iris.data, iris.target

dim_n_estimators = Integer(10, 200,name='n_estimators')
dim_max_depth = Integer(1, 20,name='max_depth')
dim_min_samples_split = Integer(2, 20,name='min_samples_split')
dim_min_samples_leaf = Integer(1, 20,name='min_samples_leaf')

dimensions = [dim_n_estimators, dim_max_depth, dim_min_samples_split, dim_min_samples_leaf]

default_params = [100,10,2,1]

rf = RandomForestClassifier(random_state=42)

def objective(**params):
    rf.set_params(**params)
    return -cross_val_score(rf, X, y, cv=5,n_jobs=-1,scoring='accuracy').mean()


search_result = gp_minimize(func=objective, dimensions=dimensions,acq_func='EI',n_calls=50,
                            x0=default_params,random_state=42)


print("最优超参数: ")
print(f"n_estimators = {search_result.x[0]}")
print(f"max_depth = {search_result.x[1]}")
print(f"min_samples_split = {search_result.x[2]}")
print(f"min_samples_leaf = {search_result.x[3]}")

print(f"\n最优交叉验证准确率: {-search_result.fun:.4f}")

from skopt.plots import plot_convergence
plot_convergence(search_result)
plt.savefig("ff.png")