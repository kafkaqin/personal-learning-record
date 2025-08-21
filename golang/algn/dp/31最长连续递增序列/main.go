package main

// dp[i]以i为结尾的最长连续递增子系列的长度为dp[i]
// dp[i] = dp[i-1]+1
func getMaxLength(nums []int) int {
	lenNums := len(nums)
	dp := make([]int, lenNums)

	for i := 0; i < lenNums; i++ {
		dp[i] = 1
	}

	result := 0
	for i := 1; i < lenNums; i++ {
		if nums[i] > nums[i-1] {
			dp[i] = dp[i-1] + 1
			result = max(result, dp[i])
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
