import math
n=60
k=12
perm = math.perm(n,k)
print(perm)

comb = math.comb(n,k)
print(comb)

def manual_perm(n,k):
    if k >n or k<0:
        return 0
    result =1
    for i in range(n,n-k,-1):
        result *= i
    return result

def manual_comb(n,k):
    if k >n or k<0:
        return 0
    c = min(k,n-k)
    numerator = 1
    denominator = 1
    for i in range(k):
        numerator *= (n-i)
        denominator *= (i+1)
    return numerator//denominator

perm_value = manual_perm(n,k)
print(perm_value)
comb_value = manual_comb(n,k)
print(comb_value)

n = 10
k = 3
roles = math.perm(n,k)
print(roles)
group =math.comb(n,k)
print(group)
red_balls = math.comb(33,6)
print(red_balls)
total = red_balls *16
print(total)

print(math.perm(5,0))
print(math.comb(5,0))
print(math.perm(5,7))
print(math.comb(5,8))
from scipy import special
print(special.perm(5,3))
print(special.comb(5,3))
print(special.comb(100,2))

from itertools import permutations,combinations

items = ['A','B','C','D']
print("所有排列")
for p in permutations(items,2):
    print(p)
print("所有组合")
for p in combinations(items,2):
    print(p)