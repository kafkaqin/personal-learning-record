package main

import (
	"sort"
)

// 先把所有的负数取反
// 把最小的整数取反
func maxSum(nums []int, k int) int {
	// 排序 按绝对值从小到大排序
	// 比较函数
	sort.Slice(nums, func(i, j int) bool {
		return abs(nums[i]) > abs(nums[j])
	})

	result := 0
	size := len(nums) - 1
	for i := 0; i <= size; i++ {
		if nums[i] < 0 && k > 0 { // 负数取反
			nums[i] *= -1
			k--
		}
		if k%2 == 1 { //最小的整数取反
			nums[size] *= -1 //绝对值最小的正数
		}
	}

	for i := 0; i < size; i++ {
		result += nums[i]
	}
	return result
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
func comp(a, b int) int {
	if a > b {
		return a
	}
	return b
}
