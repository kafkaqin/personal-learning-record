set_a = {1,2,3}
set_b = {3,4,5}
print(set_a)
print(set_b)

# union = set_a.union(set_b)
union = set_a |set_b
print(union)

intersection = set_a & set_b
print(intersection)
difference = set_a - set_b
print(difference)
difference = set_b - set_a
print(difference)

symmetric_difference = set_a ^ set_b
print(symmetric_difference)

print("A是B的子集吗:",set_a.issubset(set_b))
print("3是A的子集吗?",{3}.issubset(set_a))

print("A是{1,2}的超集吗.",set_a.issuperset({1,2}))

print("A和B相交吗",set_a.isdisjoint(set_b))

set_a.add(4)
print(set_a)

set_a.discard(4)
print(set_a)
set_a.remove(1)
print(set_a)