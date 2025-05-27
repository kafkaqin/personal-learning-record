package main

// 树层去重 树枝去重

var path []int
var result [][]int

// 排序 sum 需要对nums进行排序
func backtracing(nums []int, targetSum int, sum int, startIndex int, used []int) {
	if sum > targetSum {
		return
	}
	if sum == targetSum {
		tmp := make([]int, len(path))
		copy(tmp, path)
		result = append(result, tmp)
		return
	}

	for i := startIndex; i < len(nums); i++ {
		if i > 0 && nums[i] == nums[i-1] && used[i-1] == 0 { // 1表示已用过 0表示还未使用 树层
			continue
		}
		path = append(path, nums[i])
		sum += nums[i]
		used[i] = 1
		backtracing(nums, targetSum, sum, i, used)
		used[i] = 0
		sum -= nums[i]
		path = path[:len(path)-1]
	}
}
