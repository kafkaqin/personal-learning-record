package main

import (
	"fmt"
)

func count(s []string) []int {
	countA := make(map[string]int, 0)
	for i := range s {
		countA[fmt.Sprintf("%s", s[i])] = i
	}

	result := make([]int, 0)
	left := 0
	right := 0
	for i := 1; i < len(s); i++ {
		right = max(right, countA[s[i]])
		if right == i {
			result = append(result, right-left+1)
			left = i + 1
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
