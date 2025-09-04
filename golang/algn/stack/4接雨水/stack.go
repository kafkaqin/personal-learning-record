package main

func stacks(nums1 []int) int {
	lennums1 := len(nums1)
	if lennums1 == 0 {
		return 0
	}
	sum := 0
	stack := make([]int, lennums1)
	stack = append(stack, 0)
	for i := 1; i < lennums1; i++ {
		if nums1[i] <= nums1[stack[len(stack)-1]] { //小于等于栈顶元素 单调递增
			stack = append(stack, i) //进栈
		} else {
			for len(stack) > 0 && nums1[i] > nums1[stack[len(stack)-1]] {
				//mmiddleIndex := stack[len(stack)-1]
				stack = stack[:len(stack)-1] //出栈
				leftIndex := stack[len(stack)-1]
				left := nums1[leftIndex]
				right := nums1[i]
				hight := right - left
				kuandu := i - leftIndex - 1
				sum += hight * kuandu
			}
		}
	}
	return sum
}
