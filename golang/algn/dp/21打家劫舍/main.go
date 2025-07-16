package main

// 相邻 不能偷
// dp数组的含义: 包含下标i 偷到的最大价值为 dp ，结果放在 dp[nums.size-1]
// 字符串的长度为i，如果能组装字典中的字母就dp[i]为ture
// 偷i: dp[i-2]+nums[i]
// 不偷i: dp[i-1]
func bag(nums []int) int {

	dp := make([]int, len(nums))
	numsLen := len(nums)
	dp[0] = nums[0]
	dp[1] = max(nums[1], nums[0])
	for i := 2; i <= numsLen; i++ {
		dp[i] = 0
	}

	// 方法2 先遍历背包
	for i := 1; i < numsLen; i++ { //先遍历背包 排列数 背包就是 s 提供的字符串
		dp[i] = max(dp[i-1], dp[i-1]+nums[i])
	}

	return dp[numsLen-1]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
