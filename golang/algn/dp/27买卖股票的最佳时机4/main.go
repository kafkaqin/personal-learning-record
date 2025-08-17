package main

// dp[i][0]不操作 dp[i][0] = dp[i-1][0]
// dp[i][1]第一次持有 dp[i][1] = (dp[i-1][1],dp[i-1][0]-prices[i])
// dp[i][2]第一次不持有 dp[i][2] = (dp[i-1][2],dp[i-1][1]+prices[i])
// dp[i][3]第二次持有 dp[i][3] = (dp[i-1][3],dp[i-1][2]-prices[i])
// dp[i][4]第二次不持有 dp[i][4] = (dp[i-1][4],dp[i-1][3]+prices[i])
// dp[len(prices)-1][2k+1]

func rebTree(prices []int, k int) int {
	lenPri := len(prices)
	dp := make([][]int, len(prices))
	// j+1 持有
	// j+2 不持有

	//初始化
	dp[0][0] = 0
	for j := 0; j < 2*k; j += 2 {
		dp[0][j+1] = -prices[0] //持有
		dp[0][j+2] = 0          //不持有
	}
	for i := 1; i < lenPri; i++ {
		for j := 0; j < 2*k+1; j += 2 {
			dp[i][j+1] = 0
			dp[i][j+2] = 0
		}
	}
	// j+1 持有
	// j+2 不持有
	for i := 1; i < lenPri; i++ {
		for j := 0; j < 2*k; j += 2 {
			dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j]-prices[i])
			dp[i][j+2] = max(dp[i-1][j+2], dp[i-1][j+1]+prices[i])
		}

	}

	return dp[lenPri-1][2*k+1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
