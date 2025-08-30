package main

// 删除元素的方式
// dp[i][j] 以i到j为结尾的是否是回文子串为dp[i][j] = true或者false
// s[i]==s[j]:
// i和j相差小于等于1: if j-i <=1 { dp[i][j]==true: result++}
// i和j相差大于1: dp[i+1][j-1]==true: dp[i][j]=true;result ++
func getMaxLength(s []string) int {
	lens := len(s)
	result := 0
	dp := make([][]bool, lens)
	for i := lens - 1; i >= 0; i-- {
		for j := i; j < lens; j++ {
			if s[i] == s[j] {
				if j-i <= 1 {
					dp[i][j] = true
					result++
				} else {
					if dp[i+1][j-1] == true {
						dp[i][j] = true
						result++
					}
				}
			}
		}
	}
	return result
}
