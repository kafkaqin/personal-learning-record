a = 0.1
b = 0.2
print(a+b)

x = 0.1 + 0.1 + 0.1
y = 0.3
print(x==y)

import sys
eps = sys.float_info.epsilon
print("机器 epsilon:",eps)


sum_bad = 0.0
for _ in range(1000000):
    sum_bad += 0.1
print(sum_bad)


def kahan_sum(values):
    sum_ =  0.0
    c = 0.0
    for x in values:
        y = x - c
        t = sum_ + y
        c = (t-sum_) - y
        sum_  = t
    return sum_

values = [0.1] * 1000000
print(kahan_sum(values))

from decimal import Decimal,getcontext
getcontext().prec = 50

a = Decimal('0.1')
b = Decimal('0.2')
print(a+b)
## 分数表示形式
from fractions import Fraction
f = Fraction(1, 2)+Fraction(2, 3)
print(f)