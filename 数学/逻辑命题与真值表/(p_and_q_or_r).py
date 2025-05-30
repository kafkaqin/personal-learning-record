from itertools import product

def truth_table(variables,expr_func):
    """
    生成逻辑表达式的真值表
    :param variables :变量名列表
    :param expr_func: 接收一个变量元组，返回布尔值
    :return:
    """
    n = len(variables)
    print("\t".join(variables+["Result"]))
    for inputs in product([False,True],repeat=n):
        result = expr_func(*inputs)
        row = "\t".join([str(x) for x in inputs+(result,)])
        print(row)

def expr(p,q,r):
    return (p and q) or r

truth_table(["p","q","r"],expr)