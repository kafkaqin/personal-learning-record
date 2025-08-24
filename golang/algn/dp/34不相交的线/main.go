package main

// dp[i][j] 以i-1为结尾的nums1[i]，以j-1为结尾的nums2[j]最长公共子系列的长度
// if nums1[i-1] > nums2[j-1] {dp[i][j] = dp[i-1][j-1]+1}
func getMaxLength(nums1, nums2 []string) int {
	lennums1 := len(nums1)
	lennums2 := len(nums2)
	dp := make([][]int, lennums1+1)
	dp[0][0] = 0
	for i := 1; i <= lennums1; i++ {
		dp[i][0] = 0
	}
	for i := 1; i <= lennums2; i++ {
		dp[0][i] = 0
	}
	for i := 1; i <= lennums1; i++ {
		for j := 1; j <= lennums2; j++ {
			if nums1[i-1] > nums2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}

		}
	}
	return dp[lennums1][lennums2]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
