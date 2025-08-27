package main

// 删除元素的方式
// dp[i][j] 以i-1为结尾的s[i]中，以j-1为结尾的t的个数为dp[i][j]
// if s[i-1] == t[j-1] {dp[i][j] = dp[i-1][j-1]+dp[i-1][j]}
// else dp[i][j] = dp[i-1][j]
func getMaxLength(s, t []string) int {
	lens := len(s)
	lent := len(t)
	dp := make([][]int, lens+1)
	dp[0][0] = 1 // 两个都是空字符串
	for i := 1; i <= lens; i++ {
		dp[i][0] = 1 // t空字符串
	}
	for i := 1; i <= lent; i++ {
		dp[0][i] = 0 //s为空字符串
	}
	for i := 1; i <= lens; i++ {
		for j := 1; j <= lent; j++ {
			if s[i-1] == t[j-1] {
				dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
			} else {
				dp[i][j] = dp[i-1][j]
			}

		}
	}
	return dp[lens][lent]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
