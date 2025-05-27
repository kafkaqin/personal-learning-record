import numpy as np
data = np.array([1,2,3,4,5,6,7,8,9])
mean_value = np.mean(data)

varicance_population = np.var(data)


varicance_simple = np.var(data,ddof=1)

print("期望(均值)",mean_value)
print("总体方差(除以 n):",varicance_population)
print("样本方差(除以n-1):",varicance_simple)
import matplotlib.pyplot as plt

plt.hist(data, bins='auto', alpha=0.7, edgecolor='black')
plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(mean_value + np.std(data), color='g', linestyle='dotted', label='Mean ± Std')
plt.axvline(mean_value - np.std(data), color='g', linestyle='dotted')

plt.title('Data Distribution with Mean and Std')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig("ss.png")