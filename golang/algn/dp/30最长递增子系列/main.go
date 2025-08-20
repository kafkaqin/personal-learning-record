package main

// dp[i]以nums[i]为结尾的最长递增子系列的长度
// dp[i] = max(dp[j]+1,dp[i])
func getMaxLength(nums []int) int {
	lenNums := len(nums)
	dp := make([]int, lenNums)

	for i := 0; i < lenNums; i++ {
		dp[i] = 1
	}
	result := 0
	for i := 1; i < lenNums; i++ {
		for j := 0; j < i; j++ {
			if nums[i] > nums[j] {
				dp[i] = max(dp[i], dp[j]+1)
				result = max(result, dp[i])
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
