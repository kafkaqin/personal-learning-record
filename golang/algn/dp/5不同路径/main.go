package main

// 1. dp数组的含义 dp[i][j] 从(0,0)到(i,j)有dp[i][j]种走法
// 2. 递推公式 dp[i][j] = dp[i-1][j]+dp[i][j-1]
// 3. 如何初始化
// 4. 遍历顺序
// 5. 打印

func count(n, m int) int {
	dp := make([][]int, n)
	for i := 0; i < n; i++ {
		dp[0][i] = 1
	}
	for i := 0; i < m; i++ {
		dp[i][0] = 1
	}

	for i := 1; i < n; i++ {
		for j := 1; j < m; j++ {
			dp[i][j] = dp[i-1][j] + dp[i][j-1]
		}
	}
	return dp[n-1][m-1]
}
