package main

// 把石头分成两堆 尽可能分成重量相等的两堆
//d[j] 容量为j的最大价值
// 背包装满的条件: target = sum/2 dp[j] == target

// 一维dp[i]数组
// dp[j] 容量为j的背包所能装的最大重量为dp[j]
// dp[j] 放物品i
// 不放物品 d[j-weight[i]]+values[i]
// 递归公式:dp[j]=max(dp[j],dp[j-stones[i]]+stones[i])
// 初始化: dp[0] = 0
// 遍历顺序
// 打印dp数组
func bag01(stones []int) int {
	sum := 0
	for _, v := range stones {
		sum += v
	}

	dp := make([]int, len(stones))
	dp[0] = 0
	for i := 0; i < len(stones); i++ {
		dp[i] = 0
	}
	target := sum / 2
	for i := 1; i < len(stones); i++ { //物品
		for j := target; j >= stones[i]; j-- { //背包 每个物品只能使用一次 所以使用一次
			dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
		}
	}

	return abs(sum - dp[target])
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 面试题目: 两个for 循环可以不可以倒叙;为什么一维dp数组不可以倒叙
