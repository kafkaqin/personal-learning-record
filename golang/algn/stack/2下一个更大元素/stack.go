package main

func stacks(nums1, nums2 []int) []int {
	lennums1 := len(nums1)
	if lennums1 == 0 {
		return []int{}
	}
	lennums2 := len(nums2)
	stack := make([]int, lennums2)
	nums1Map := make(map[int]int)
	for i := 0; i < lennums1; i++ {
		nums1Map[nums1[i]] = i
	}

	result := make([]int, len(nums1))
	stack = append(stack, 0)
	for i := 1; i < lennums2; i++ {

		if nums2[i] <= nums2[stack[len(stack)-1]] { //小于等于栈顶元素 单调递增
			stack = append(stack, i) //进栈
		} else {
			for len(stack) > 0 && nums2[i] > nums2[stack[len(stack)-1]] {
				num1Index, ok := nums1Map[nums2[i]]
				if ok {
					result[num1Index] = nums2[i]
				}
				stack = stack[:len(stack)-1] //出栈
			}
		}
	}
	return result
}
