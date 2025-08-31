package main

// dp[i][j] 以[i,j]为范围的最长回文子串的长度为dp[i][j]
// s[i]==s[j]:
//
//	if s[j]==s[i]  { dp[i][j]=dp[i+1][j-1] +2}
//
// else dp[i][j]=max(dp[i][j-1],dp[i+1][j])
func getMaxLength(s []string) int {
	lens := len(s)
	dp := make([][]int, lens)
	for i := 0; i < lens; i++ {
		dp[i][i] = 1
	}
	for i := lens - 1; i >= 0; i-- {
		for j := i + 1; j < lens; j++ {
			if s[i] == s[j] {
				dp[i][j] = dp[i+1][j-1] + 2
			} else {
				dp[i][j] = max(dp[i][j-1], dp[i+1][j])
			}
		}
	}
	return dp[0][lens-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
