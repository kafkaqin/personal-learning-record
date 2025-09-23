package main

var path []int
var result [][]int

// 排序 sum
func backtracing(nums []int, startIndex int) {
	if len(path) >= 2 {
		tmp := make([]int, len(path))
		copy(tmp, path)
		result = append(result, tmp)
		return
	}

	//set
	set := make(map[int]bool)
	for i := startIndex; i < len(nums); i++ {
		if set[nums[i]] {
			continue
		}
		if len(path) > 0 && path[len(path)-1] > nums[i] {
			continue
		}
		set[nums[i]] = true //只负责本层的去重
		path = append(path, nums[i])
		backtracing(nums, i+1)
		path = path[:len(path)-1]
		//set[nums[i]] = false

	}
}
