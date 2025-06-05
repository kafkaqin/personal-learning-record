"""
快速排序基本思想：
选择一个“基准值”（pivot）
将小于 pivot 的元素放到左边，大于的放到右边（分区）
对左右两部分递归地进行快速排序
"""

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    right = [x for x in arr[1:] if x >pivot]
    return quick_sort(left) + [pivot] + quick_sort(right)

# 示例：
arr = [3, 6, 8, 10, 1, 2, 1]
print("原始数组:", arr)
sorted_arr = quick_sort(arr)
print("排序后:", sorted_arr)