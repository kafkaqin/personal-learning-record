package main

import "math"

// 装满容量为i背包，最少物品为dp[i]
// dp[j]  = min(dp[j-coins[i]]+1,dp[j])
func bag(amount int, coins []int) int {
	dp := make([]int, len(coins))
	coinsLen := len(coins)
	dp[0] = 0
	for i := 1; i <= coinsLen; i++ {
		dp[i] = math.MaxInt32
	}
	// 方法1 先遍历物品
	for i := 0; i < coinsLen; i++ { //先遍历物品  组合数
		for j := coins[i]; j <= amount; j++ { //再遍历背包
			dp[j] = min(dp[j], dp[j-coins[i]]+1)
		}
	}
	// 方法2 先遍历背包
	for j := 0; j <= amount; j++ { //先遍历背包 排列数
		for i := 0; i < coinsLen; i++ { //后遍历物品
			dp[j] = min(dp[j], dp[j-coins[i]]+1)
		}
	}

	return dp[amount]
}

func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
