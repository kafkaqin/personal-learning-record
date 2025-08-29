def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_memo(n,memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    else:
        memo[n] = fibonacci(n-1) + fibonacci(n-2)
        return memo[n]

fib = lambda n:n if n <=1 else  fib(n-1)+fib(n-2)


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)

def quicksort_inplace(arr,low=0,high=None):
    if high == None:
        high = len(arr)-1
    if low < high:
        pivot_index = partition(arr,low,high)
        quicksort_inplace(arr,low,pivot_index-1)
        quicksort_inplace(arr,pivot_index+1,high)

def partition(arr,low,high):
    pivot = arr[high]
    i = low-1
    for j in range(low,high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1