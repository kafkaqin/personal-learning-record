package main

var path []int
var result [][]int

/*
*
组合: 不强调顺序
排列: 顺序相关
*/
func backtracing(nums []int, used []int) {
	if len(nums) == len(path) {
		tmp := make([]int, len(path))
		copy(tmp, path)
		result = append(result, tmp)
		return
	}
	for i := 0; i < len(nums); i++ { //排序
		if used[i] == 1 {
			continue
		}
		used[i] = 1
		path = append(path, nums[i])
		backtracing(nums, used)
		path = path[:len(path)-1]
		used[i] = 0
	}
}
