package main

// dp[i] 以i为下标 num[i]结尾的最大子序和
// dp[i] = max(dp[i-1]+nums[i],nums[i])
func getMaxLength(nums []int) int {
	lennums := len(nums)
	dp := make([]int, lennums)
	dp[0] = nums[0]

	result := 0
	for i := 1; i < lennums; i++ {
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		result = max(result, dp[i])
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
