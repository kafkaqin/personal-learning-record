set_a = {1,2,3}
set_b = {3,4}

print("1交集: ",set_a & set_b)
print("2交集: ",set_a.intersection(set_b)) # 交集
# 并集
print("1并集: ",set_a | set_b)
print("2并集: ",set_a.union(set_b))

# 差集

print("1差集: ",set_a - set_b)
print("2差集: ",set_a.difference(set_b))

# 对称差集
print("1对称差集: ",set_a ^ set_b)
print("2对称差集: ",set_a.symmetric_difference(set_b))

# 子集 与 超集的判断
set_c = {1,2}
print("子集: ",set_c.issubset(set_a))
print("超集: ",set_a.issuperset(set_c))