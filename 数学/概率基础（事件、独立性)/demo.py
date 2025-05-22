import numpy as np
import matplotlib.pyplot as plt
# n = 10
# p = 0.5
#
# result = np.random.binomial(n,p)
#
# print("在",n,"次抛硬币，出现了",result,"次正面")
trials = 1000
n = 10
p = 0.5

result = np.random.binomial(n,p,trials)

plt.hist(result,bins=np.arange(-0.5,n+1.5,1),rwidth=0.8,edgecolor='black')
plt.title("Binomial Distribution: 10 coin flips, p=0.5")
plt.xlabel("Number of Heads")
plt.ylabel("Frequency")
plt.xticks(range(n+1))
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.savefig("test.png")