package main

// 覆盖范围
func isCover(nums []int) bool {
	if len(nums) == 1 {
		return true
	}
	cover := 0
	size := len(nums) - 1
	for i := 0; i <= cover; i++ {
		cover = max(i+nums[i], cover) //更新覆盖范围 尽可能增加覆盖范围
		if cover >= size {
			return true
		}
	}
	return false
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
