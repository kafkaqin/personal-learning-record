import numpy as np

data = [88,56,33,2,2,221,112,11,1,11,11]

mean_data_manual = sum(data)/len(data)
print("手动==",mean_data_manual)
var_data_manual = sum((x-mean_data_manual)**2 for x in data)/len(data)
print("手动==",var_data_manual)
data = np.array(data)
print(data)
mean_data = np.mean(data)
print(mean_data)
var_data = np.var(data)
print(var_data)

var_data = np.var(data,ddof=1)
print(var_data)

std_data = np.std(data)
print(std_data)

std_data = np.std(data,ddof=1)
print(std_data)



# 二维数据：每行是一个学生的多门成绩
scores = np.array([
    [80, 85, 78],
    [90, 92, 88],
    [75, 80, 70],
    [88, 90, 91]
])

print("原始数据（学生 × 科目）:")
print(scores)

# 按列（每科）计算均值
mean_per_subject = np.mean(scores, axis=0)
print(f"各科目平均分: {mean_per_subject}")

# 按行（每个学生）计算方差
var_per_student = np.var(scores, axis=1, ddof=1)  # 每个学生的成绩波动
print(f"每个学生的成绩方差: {var_per_student}")