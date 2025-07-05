package main

// 放入物品i 容量为j的背包的最大价值为dp[i][j]
// dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight]+values[i])

// 一维dp[i]数组
// dp[j] 容量为j的背包所能装的最大价值为dp[j]
// 递归公式:max(dp[j],dp[j-weight[i]]+values[i])
// 初始化: dp[0] = 0
// 遍历顺序
// 打印dp数组
func bag01(n int, bagweight int, weight []int, values []int) int {
	dp := make([]int, 10)
	dp[0] = 0
	for i := 0; i < n; i++ { //物品
		for j := bagweight; j >= weight[i]; j-- { //背包
			dp[j] = max(dp[j], dp[j-weight[i]]+values[i])
		}
	}
	return dp[bagweight]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 面试题目: 两个for 循环可以不可以倒叙;为什么一维dp数组不可以倒叙
