package main

// 字符串的长度为i，如果能组装字典中的字母就dp[i]为ture
// if[j,i]==wordDict[] &&dp[j] == true
// dp[i] = true
func bag(s string, words []string) bool {
	wordDict := make(map[string]struct{})
	for _, w := range words {
		wordDict[w] = struct{}{}
	}
	dp := make([]bool, len(s))
	sLen := len(s)
	dp[0] = true
	for i := 1; i <= sLen; i++ {
		dp[i] = false
	}

	// 方法2 先遍历背包
	for i := 1; i < sLen; i++ { //先遍历背包 排列数 背包就是 s 提供的字符串
		for j := 0; j < i; j++ { //后遍历物品
			word := s[j:i]
			if _, ok := wordDict[word]; ok && dp[j] {
				dp[i] = true
			}
		}
	}

	return dp[sLen-1]
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
