import math

def perm(n, k):
    return math.factorial(n) // math.factorial(n - k)

def comb(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

print(perm(5, 3))  # 60
print(comb(5, 3))  # 10