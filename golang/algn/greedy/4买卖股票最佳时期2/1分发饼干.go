package main

import "math"

func greety(prices []int) int {
	var result = math.MinInt
	for i := 1; i < len(prices); i++ {
		sum := prices[i] - prices[i-1]
		if sum > 0 {
			result += sum
		}
	}
	return result
}
