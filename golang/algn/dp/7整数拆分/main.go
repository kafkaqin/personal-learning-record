package main

// 1. dp数组的含 dp[i] 对i进行拆分得到的最大的乘积为dp[i]
// 2. 递推公式 dp[i] = max(j*dp[i-j],j*(i-j))
// 3. 如何初始化
// 4. 遍历顺序
// 5. 打印

/*
0 表示无障碍
1 有障碍
*/
func count(n int) int {

	dp := make([]int, n)
	dp[0] = 0
	dp[1] = 1
	dp[2] = 2
	for i := 3; i <= n; i++ {
		for j := 1; j < i; j++ { // 对 i 进行拆分
			tmp := max(dp[i], j*dp[i-j])
			dp[i] = max(j*(i-j), tmp)
		}
	}
	return dp[n]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
