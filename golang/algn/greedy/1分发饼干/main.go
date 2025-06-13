package main

// 先排序

func greety(g []int, s []int) int {
	result := 0
	index := len(s) - 1
	for i := len(g); i >= 0; i-- { //先遍历胃口
		for (index >= 0) && (s[index] >= g[i]) { //再遍历饼干
			result = result + 1
			index--
		}
	}
	return result
}
