import numpy as np

def estimate_pi(n_samples):
    points = np.random.uniform(-1,1,size=(n_samples,2))

    distances_squared = np.sum(points ** 2, axis=1)

    inside_circle = np.sum(distances_squared<=1)

    pi_estimate = 4* inside_circle/n_samples
    return pi_estimate

def estimate_integral(n_samples):
    points = np.random.rand(n_samples,2)
    inside_area = np.sum(points[:,1]<=points[:,0]**2)
    integral_estimate = inside_area/n_samples
    return integral_estimate
if __name__ == '__main__':
    n = 1_000_000
    pi = estimate_pi(n)
    print(f"估计 pi 值({n}次实验): ",pi)

    integral=estimate_integral(n)
    print(f"估计积分",integral)