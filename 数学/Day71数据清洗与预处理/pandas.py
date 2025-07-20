import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


data = {
    'Age' :[25,np.nan,35,28,np.nan,40],
    'Salary': [50000,62000,np.nan,58000,60000,70000],
    'Department' : ['HR','IT',np.nan,'IT','HR','Finance']
}

df = pd.DataFrame(data)
print("原始数据:")
print(df)

df_filled = df.copy()
df_filled[['Age','Salary']] = df_filled[['Age','Salary']].fillna(df[['Age','Salary']].mean())

print("\n 填充后的数据:")
print(df_filled)

df_dropped = df.dropna()
print("\n删除缺失值后的数据:")
print(df_dropped)

scaler = StandardScaler()
df_scaled = df_filled.copy()
df_scaled[[
    'Age','Salary'
]] = scaler.fit_transform(df_scaled[['Age','Salary']])

print("\n 标准化后的数据:")
print(df_scaled)