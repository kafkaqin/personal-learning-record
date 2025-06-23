package main

import "sort"

func count(nums [][]int) [][]int {
	if len(nums) == 0 {
		return [][]int{}
	}

	sort.Slice(nums, func(i, j int) bool {
		return nums[i][0] < nums[j][0] //按从左边界从小到大排序
	})

	result := make([][]int, 0)
	result = append(result, nums[0])
	for i := 1; i < len(nums); i++ {
		pre := result[len(result)-1][1]
		if nums[i][0] <= pre { //重叠
			//合并
			pre = max(nums[i][1], pre)

		} else {
			result = append(result, nums[i])
		}
	}

	return result
}

func max(a, b int) int {
	if a < b {
		return b
	}
	return b
}
