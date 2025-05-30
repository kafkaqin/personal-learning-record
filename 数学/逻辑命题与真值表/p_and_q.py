'''
生成 p and q
'''
values = [(p,q) for p in [False,True] for q in [False,True]]
# print(values)
print("p\tq\tp and q") ###
for p,q in values:
    result = p and q
    print("{}\t{}\t{}".format(p,q,result))