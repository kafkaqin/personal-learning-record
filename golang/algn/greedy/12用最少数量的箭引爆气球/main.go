package main

import "sort"

// 1. 按身高排序
// 2. 按 k 排序
func count(ponds [][]int) int {
	if len(ponds) == 0 {
		return 0
	}

	sort.Slice(ponds, func(i, j int) bool {
		return ponds[i][0] < ponds[j][0] //按从左边界从小到大排序
	})

	result := 1 // 至少需要一个弓箭
	for i := 1; i < len(ponds); i++ {
		if ponds[i][0] > ponds[i-1][1] {
			result++
		} else {
			ponds[i][1] = min(ponds[i][1], ponds[i-1][1]) //更新右边界 (方便判断下一个的左边界和当前气球的右边界是否重叠)
		}
	}

	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
