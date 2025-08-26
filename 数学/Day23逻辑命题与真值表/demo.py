values = [(p,q) for p in [True, False] for q in [True, False]]
print("p\tq\t\p and q \t p or q\ not p\t ^p(xor)")
print("-"*25)
for p,q in values:
    and_result = p and q
    or_result = p or q
    not_p = not p
    xor_result = p^q
    print(f"{p}\t{q}\t{and_result}\t{or_result}\t{not_p}\t{xor_result}")
print("-"*25)

print("-"*25)
from itertools import product
variables = ['p','q','r']
print("\t".join(variables)+"\t(p and q) or not r")
print("-"*50)

for p,q,r in product([False,True],repeat=3):
    result = (p and q) or (not r)
    print(f"{p}\t{q}\t{r}\t{result}")


def truth_table(variables, expr_func, expr_name="expr"):
    """
    生成逻辑表达式的真值表
    :param variables: 变量名列表，如 ['p', 'q']
    :param expr_func: 接收变量并返回布尔结果的函数
    :param expr_name: 表达式名称
    """
    n = len(variables)
    print("\t".join(variables) + f"\t{expr_name}")
    print("-" * (len(variables) * 8 + len(expr_name) + 1))

    for values in product([False, True], repeat=n):
        result = expr_func(*values)
        values_str = "\t".join(str(v) for v in values)
        print(f"{values_str}\t{result}")


# 使用示例：(p → q) 等价于 (not p) or q
def implication(p, q):
    return (not p) or q
print("p\tq\tnot(p or q)\t(not p) and (not q)\t等价?")
print("-" * 70)
for p, q in product([False, True], repeat=2):
    left = not (p or q)
    right = (not p) and (not q)
    equivalent = left == right
    print(f"{p}\t{q}\t{left}\t\t{right}\t\t\t{equivalent}")

truth_table(['p', 'q'], implication, 'p → q (implies)')