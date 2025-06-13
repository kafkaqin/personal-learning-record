package main

import "math"

func greety(nums []int) int {
	var result = math.MinInt
	count := 0
	for i := 0; i <= len(nums); i++ {
		count += nums[i]
		if count > result {
			result = count
		}
		if count < 0 {
			count = 0
		}
	}
	return result
}
