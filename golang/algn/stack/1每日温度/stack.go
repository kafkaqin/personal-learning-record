package main

func stacks(T []int) []int {
	stack := make([]int, len(T))
	result := make([]int, len(T))
	stack = append(stack, 0)
	for i := 1; i < len(T); i++ {
		if T[i] <= T[stack[len(stack)-1]] { //小于等于栈顶元素
			stack = append(stack, i) //进栈
		} else { //T[i] > T[stack[len(stack)-1]
			for len(stack) > 0 && T[i] > T[stack[len(stack)-1]] {
				result[i] = i - stack[len(stack)-1]
				stack = stack[:len(stack)-1] //出栈
			}
		}
	}
	return result
}
