package main

// 收尾相连
// 相邻 不能偷
// dp数组的含义: 包含下标i 偷到的最大价值为 dp ，结果放在 dp[nums.size-1]
// 偷i: dp[i-2]+nums[i]
// 不偷i: dp[i-1]
func bag(nums []int) int {

	// 1.不考虑 首尾
	// 2.考虑首元数
	// 3.考虑尾元数
	dp := make([]int, len(nums))
	numsLen := len(nums)
	dp[0] = nums[0]
	dp[1] = max(nums[1], nums[0])
	for i := 2; i <= numsLen; i++ {
		dp[i] = 0
	}

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

func main() {
	// 2.考虑首元数
	// 3.考虑尾元数
	nums := []int{}
	a := bag(nums[1:])
	b := bag(nums[:len(nums)-1])
	max(a, b)
}
