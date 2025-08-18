package main

// dp[i][0]持有股票  dp[i][1] 保持卖出股票的状态 dp[i][2] 具体卖出股票  dp[i][3] 冷冻期
// dp[i][0]: max(dp[i-1][0],dp[i-1][3]-prices[i](冷冻期后买入),dp[i-1][1]-prices[i](买入))
// dp[i][1]: max(dp[i-1][1],dp[i-1][3])
// dp[i][2]: dp[i-1][0]+prices[i]
// dp[i][3] = dp[i-1][2]
func rebTree(prices []int) int {
	lenPri := len(prices)
	dp := make([][]int, len(prices))
	dp[0][0] = -prices[0]
	dp[0][1] = 0
	dp[0][2] = 0
	dp[0][3] = 0
	for i := 1; i < lenPri; i++ {
		dp[i][0] = 0
		dp[i][1] = 0
		dp[i][2] = 0
		dp[i][3] = 0
	}

	for i := 1; i < lenPri; i++ {
		dp[i][0] = max(max(dp[i-1][0], dp[i-1][3]-prices[i]), dp[i-1][1]-prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][3])
		dp[i][2] = dp[i-1][0] + prices[i]
		dp[i][3] = dp[i-1][2]
	}

	return max(max(dp[lenPri-1][1], dp[lenPri-1][2]), dp[lenPri-1][3])
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
