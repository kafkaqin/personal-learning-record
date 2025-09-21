package main

var path []int
var result [][]int

// 排序 sum
func backtracing(nums []int, startIndex int) {
	result = append(result, path)
	if len(nums) <= startIndex {
		return
	}
	for i := startIndex; i < len(nums); i++ {
		path = append(path, nums[i])
		backtracing(nums, i+1)
		path = path[:len(path)-1]
	}
}
