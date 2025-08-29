package main

// 删除元素的方式
// dp[i][j] 使以i-1为结尾的world1[i]中，以j-1为结尾的world2相同的最少删除操作数为dp[i][j]
// if world1[i-1] == world2[j-1] {dp[i][j] = dp[i-1][j-1]}
// else dp[i][j] = (dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+1)
func getMaxLength(world1, world2 []string) int {
	lenworld1 := len(world1)
	lenworld2 := len(world2)
	dp := make([][]int, lenworld1+1)
	dp[0][0] = 0 // 两个都是空字符串
	for i := 1; i <= lenworld1; i++ {
		dp[i][0] = i // t空字符串
	}
	for i := 1; i <= lenworld2; i++ {
		dp[0][i] = i //s为空字符串
	}
	for i := 1; i <= lenworld1; i++ {
		for j := 1; j <= lenworld2; j++ {
			if world1[i-1] == world2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1)
				dp[i][j] = min(dp[i-1][j-1]+1, dp[i][j])
			}

		}
	}
	return dp[lenworld1][lenworld2]
}

func min(a, b int) int {
	if a > b {
		return a
	}
	return b
}
