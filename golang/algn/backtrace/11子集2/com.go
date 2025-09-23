package main

var path []int
var result [][]int

// 排序 sum
func backtracing(nums []int, startIndex int, used []bool) {
	result = append(result, path)
	if len(nums) <= startIndex {
		return
	}
	for i := startIndex; i < len(nums); i++ {
		if nums[i] == nums[i-1] && used[i-1] == false { //树层去重
			continue
		}
		used[i] = true
		path = append(path, nums[i])
		backtracing(nums, i+1, used)
		path = path[:len(path)-1]
		used[i] = false

	}
}
