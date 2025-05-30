def logical_expression(p,q):
    return  p or not q


values2 = [(p,q) for p in [False,True] for q in [False,True]]
print("p\tq\tp or not q")

for p,q in values2:
    result = p or not q ## 异或
    print("{}\t{}\t{}".format(p,q,result))