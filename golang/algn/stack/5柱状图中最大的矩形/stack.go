package main

func stacks(nums1 []int) int {
	lennums1 := len(nums1)
	if lennums1 == 0 {
		return 0
	}
	nums1 = append(nums1, 0)           //后面加0
	nums1 = append([]int{0}, nums1...) //头部加上0
	result := 0
	stack := make([]int, lennums1)
	stack = append(stack, 0)
	for i := 1; i < lennums1; i++ {
		if nums1[i] >= nums1[stack[len(stack)-1]] { //小于等于栈顶元素 单调递增
			stack = append(stack, i) //进栈
		} else {
			for len(stack) > 0 && nums1[i] < nums1[stack[len(stack)-1]] {
				mmiddleIndex := stack[len(stack)-1]
				stack = stack[:len(stack)-1] //出栈
				if len(stack) == 0 {
					break
				}
				rightIndex := stack[len(stack)-1]
				hight := nums1[mmiddleIndex]
				kuandu := rightIndex - i - 1
				result = max(hight*kuandu, result)
			}
		}
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
