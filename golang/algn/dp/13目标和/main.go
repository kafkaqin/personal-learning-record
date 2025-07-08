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
// left 正数的集合
// right负数的集合
// left + right = sum
// left -right = target
// right = sum - left
// left - (sum-left) = target ===>>
// left=(target+sum)2

// dp[j] : 装满容量为j有dp[j]种方法
// dp[j-nums[i]]
// dp[j] +=dp[j-nums[i]]
func bag01(stones []int, target int) int {
	sum := 0
	for _, v := range stones {
		sum += v
	}

	dp := make([]int, len(stones))
	dp[0] = 1 // 装满容量为0公有dp[0] 方法 1
	for i := 1; i < len(stones); i++ {
		dp[i] = 0
	}
	bagsize := (sum + target) / 2
	for i := 1; i < len(stones); i++ { //物品
		for j := bagsize; j >= stones[i]; j-- { //背包 每个物品只能使用一次 所以使用一次
			dp[j] += dp[j-stones[i]]
		}
	}

	return dp[target]
}
