from scipy.optimize import linprog

c = [-3,-5]

A_ub = [
    [3,2],
    [1,0],
    [0,1],
]

b_ub = [18,4,6]

bounds = [(0,None),(0,None)]

result = linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs')

if result.success:
    print("成功找到最优解")
    print("最大利润:",-result.fun)
    print("最优产量 x1 =",result.x[0])
    print("最优产量 x2 =",result.x[1])
else:
    print("没有找到可行解")