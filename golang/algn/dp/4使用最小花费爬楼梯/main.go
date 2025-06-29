package main

// 可以走1或者2阶

// 1. dp数组的含义
// 2. 递推公式 dp[i] = dp[i-1]+dp[i-2]
// 3. 如何初始化
// 4. 遍历顺序
// 5. 打印

// n=1 1
// n=2 2(1,1) (1,2)
// n=3 1+2=3
// n=4 3+2
func paloudi(n int, cost []int) int {
	//dp数组的含义: 达到第i阶所需要的花费为dp[i]
	// dp[i]= dp[i-1]+cost[i-1]
	// dp[i]= dp[i-2]+cost[i-2]
	// dp[i] = min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])

	dp := make([]int, n+1)
	dp[0], dp[1] = 0, 0
	for i := 2; i <= n; i++ {
		dp[i] = min(dp[i-1]+cost[i-1], dp[i-2]+cost[i-2])
	}
	return dp[n+1]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
