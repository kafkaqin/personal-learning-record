package main

// 1. dp数组的含义
// 2. 递推公式 dp[i] = dp[i-1]+dp[i-2]
// 3. 如何初始化
// 4. 遍历顺序
// 5. 打印

func fabic(n int) int {
	dp := make([]int, n+1)
	dp[0], dp[1], dp[2] = 1, 1, 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
