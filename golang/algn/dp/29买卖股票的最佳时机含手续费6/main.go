package main

// dp[i][0]持有股票最大值  dp[i][1] 不持有股票最大值
// dp[i][0] = max(dp[i-1][0],dp[i-1][1]-prices[i])
// dp[i][1] = max(dp[i-1][1],dp[i-1][0]+prices[i]-2) //减掉手续费
func rebTree(prices []int) int {
	lenPri := len(prices)
	dp := make([][]int, len(prices))
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	for i := 1; i < lenPri; i++ {
		dp[i][0] = 0
		dp[i][1] = 0
	}

	for i := 1; i < lenPri; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i]-2)
	}

	return max(dp[lenPri-1][0], dp[lenPri-1][1])
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
