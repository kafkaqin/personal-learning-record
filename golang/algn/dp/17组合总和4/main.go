package main

// 装满容量为i 有dp[i]种方法
// dp[j] += dp[j-coins[i]]
func bag(n int, coins []int) int {
	dp := make([]int, len(coins))
	coinsLen := len(coins)
	dp[0] = 1
	for i := 1; i <= coinsLen; i++ {
		dp[i] = 0
	}
	//for i := 0; i < coinsLen; i++ { //先遍历物品  组合数
	//	for j := coins[i]; j <= n; j++ { //再遍历背包
	//		dp[j] += dp[j-coins[i]]
	//	}
	//}

	for j := 0; j <= n; j++ { //先遍历背包 排列数
		for i := 0; i < coinsLen; i++ { //后遍历物品
			dp[j] += dp[j-coins[i]]
		}
	}

	return dp[n]
}
