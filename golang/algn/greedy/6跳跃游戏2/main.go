package main

// 覆盖范围
func isCover(nums []int) int {
	if len(nums) == 1 {
		return 0
	}
	curCover := 0
	nextCover := 0
	result := 0
	size := len(nums) - 1
	for i := 0; i <= size; i++ {
		nextCover = max(i+nums[i], nextCover) //更新覆盖范围 尽可能增加覆盖范围
		if i == curCover {
			if curCover < size { //当前的覆盖范围没有达到终点
				result++             //记录步数
				curCover = nextCover // 启用下一步覆盖范围
				if curCover >= size {
					break
				}
			} else {
				break
			}
		}
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
