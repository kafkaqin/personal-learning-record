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
func paloudi(n int) int {
	//dp数组的含义: 达到第i阶有dp[i]种方法
	dp := make([]int, n+1)
	dp[0], dp[1], dp[2] = 1, 1, 2
	for i := 3; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
