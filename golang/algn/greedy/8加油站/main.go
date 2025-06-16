package main

func maxSum(gos []int, cos []int) int {
	totalSum := 0
	size := len(gos) - 1
	startIndex := 0
	curSum := 0
	for i := 0; i <= size; i++ {
		diff := gos[i] - cos[i]
		totalSum += diff
		curSum += diff
		if curSum < 0 {
			startIndex = i + 1
			curSum = 0
		}

	}
	if totalSum < 0 {
		return -1
	}
	return startIndex
}
