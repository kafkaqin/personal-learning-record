package main

import "sort"

func count(nums [][]int) int {
	if len(nums) == 0 {
		return 0
	}

	sort.Slice(nums, func(i, j int) bool {
		return nums[i][0] < nums[j][0] //按从左边界从小到大排序
	})

	result := 1 // 至少需要一个弓箭
	for i := 1; i < len(nums); i++ {
		if nums[i][0] >= nums[i-1][1] { //不重叠
		} else {
			result++
			nums[i][1] = min(nums[i][1], nums[i-1][1]) //更新右边界 (方便判断下一个的左边界和当前气球的右边界是否重叠)
		}
	}

	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
