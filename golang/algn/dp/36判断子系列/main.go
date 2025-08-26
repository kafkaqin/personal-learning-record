package main

// dp[i][j] 以i-1为结尾的str1[i]，以j-1为结尾的str2[j]最长子系列的长度
// if str1[i-1] == str2[j-1] {dp[i][j] = dp[i-1][j-1]+1}
func getMaxLength(str1, str2 []string) int {
	lenstr1 := len(str1)
	lenstr2 := len(str2)
	dp := make([][]int, lenstr1+1)
	dp[0][0] = 0
	for i := 1; i <= lenstr1; i++ {
		dp[i][0] = 0
	}
	for i := 1; i <= lenstr2; i++ {
		dp[0][i] = 0
	}
	for i := 1; i <= lenstr1; i++ {
		for j := 1; j <= lenstr2; j++ {
			if str1[i-1] > str2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}

		}
	}
	return dp[lenstr1][lenstr2]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
