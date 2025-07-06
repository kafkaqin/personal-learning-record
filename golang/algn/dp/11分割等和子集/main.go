package main

// 可以使用回溯算法 2的n次方
//d[j] 容量为j的最大价值
// 背包装满的条件: target = sum/2 dp[j] == target

// 一维dp[i]数组
// dp[j] 容量为j的背包所能装的最大价值为dp[j]
// dp[j] 放物品i
// 不放物品 d[j-weight[i]]+values[i]
// 递归公式:max(dp[j],dp[j-nums[i]]+nums[i])
// 初始化: dp[0] = 0
// 遍历顺序
// 打印dp数组
func bag01(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 0
	for i := 0; i < len(nums); i++ {
		dp[i] = 0
	}
	for i := 1; i < len(nums); i++ { //物品
		for j := len(nums) / 2; j >= nums[i]; j-- { //背包 每个物品只能使用一次 所以使用一次
			dp[j] = max(dp[j], dp[j-nums[i]]+nums[i])
		}
	}
	return dp[len(nums)-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 面试题目: 两个for 循环可以不可以倒叙;为什么一维dp数组不可以倒叙
