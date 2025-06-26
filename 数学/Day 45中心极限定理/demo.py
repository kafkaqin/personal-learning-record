import numpy as np
import matplotlib.pyplot as plt

sample_size = 30
num_samples = 1000
distribution = 'uniform'

if distribution == 'uniform':
    data = [np.mean(np.random.uniform(9,1,sample_size))  for _ in range(num_samples)]
elif distribution == 'exponential':
    data = [np.mean(np.random.exponential(1.0,sample_size)) for _ in range(num_samples)]

plt.hist(data,bins=25,density=True,alpha=0.6,color='g')

mu,sigma = np.mean(data),np.std(data)
xmin,xmax = plt.xlim()
x = np.linspace(xmin,xmax,100)
p = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

plt.plot(x,p,'k',linewidth=2)
title = f"Sample Means Distribution (Simple Size={sample_size},Samples={num_samples})"
plt.title(title)
plt.xlabel("Mean Value")
plt.ylabel("Density")
plt.grid(True)
plt.savefig("plot.png")

print(f"Sample Mean:{mu}")
print(f"Standard Deviation of Sample Means:{sigma}")