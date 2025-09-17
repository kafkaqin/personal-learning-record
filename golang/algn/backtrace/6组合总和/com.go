package main

import "strconv"

var path []string
var result [][]string

// 排序 sum
func backtracing(nums []int, target int, sum int, startIndex int) {
	if sum > target {
		return
	}
	if sum == target {
		tmp := make([]string, len(path))
		copy(tmp, path)
		result = append(result, tmp)
		return
	}

	for i := startIndex; i < len(nums); i++ {
		path = append(path, strconv.Itoa(nums[i]))
		sum += nums[i]
		backtracing(nums, target, sum, i)
		sum -= nums[i]
		path = path[:len(path)-1]
	}
}
