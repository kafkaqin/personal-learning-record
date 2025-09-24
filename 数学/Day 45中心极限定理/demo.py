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



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)

n_simulations = 50000
sample_sizes = [1,2,5,10,30,100]
population_dist = 'exponential'
lambda_param = 2.0
p_param = 0.3

if population_dist == 'exponential':
    mu = 1/lambda_param
    sigma = 1/lambda_param
elif population_dist == 'uniform':
    a,b = 0,1
    mu = (a+b)/2
    sigma = (b-a)/np.sqrt(12)
elif population_dist == 'poisson':
    mu = lambda_param
    sigma = np.sqrt(lambda_param)
elif population_dist == 'binomial':
    n_binom = 10
    mu = n_binom * p_param
    sigma = np.sqrt(n_binom * p_param*(1-p_param))

def generate_sample_mean(n,dist,params):
    if dist == 'exponential':
        sample = np.random.exponential(scale=1/params['lambda'],size=n)
    elif dist == 'uniform':
        sample = np.random.uniform(low=params['a'],high=params['b'],size=n)
    elif dist == 'poisson':
        sample = np.random.poisson(lam=params['lambda'],size=n)
    elif dist == 'binomial':
        sample = np.random.binomial(n=params['n'],p=params['p'],size=n)
    return np.mean(sample)


fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(15,10))
axes = axes.ravel()
for idx, n in enumerate(sample_sizes):
    # 存储每次模拟的样本均值
    sample_means = np.zeros(n_simulations)

    # 参数字典
    params = {
        'lambda': lambda_param,
        'a': 0, 'b': 1,
        'n': 10, 'p': p_param
    }

    # 抽样模拟
    for i in range(n_simulations):
        sample_means[i] = generate_sample_mean(n, population_dist, params)

    # 标准化样本均值
    z_scores = (sample_means - mu) / (sigma / np.sqrt(n))

    # 绘制直方图（密度）
    ax = axes[idx]
    ax.hist(z_scores, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label=f'n = {n}')

    # 绘制标准正态分布曲线
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')

    ax.set_title(f'样本量 n = {n}')
    ax.set_xlabel('标准化样本均值 $Z_n$')
    ax.set_ylabel('密度')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle(f'中心极限定理模拟验证（总体分布：{population_dist}）', fontsize=16)
plt.tight_layout()
plt.show()