def factorial(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return factorial(n-1)+factorial(n-2)

for i in range(1,11):
    print(factorial(i))