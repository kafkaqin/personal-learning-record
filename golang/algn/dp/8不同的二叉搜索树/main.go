package main

//二叉搜索树
// 左子树
// 1. dp数组的含 dp[i] 对i进行分解成为dp[i]种不同的二叉搜索树
// 2. 递推公式 dp[i]+=dp[j-1]*dp[i-j]         dp[3]= dp[0]*dp[2]+dp[1]*dp[1]+dp[2]*dp[0]
// 3. 如何初始化
// 4. 遍历顺序
// 5. 打印

func count(n int) int {

	dp := make([]int, n)
	dp[0] = 1 // 空二叉树
	for i := 1; i <= n; i++ {
		dp[i] = 0
	}
	for i := 1; i <= n; i++ {
		for j := 1; j < i; j++ { // 以 j 为头节点的情况
			dp[i] += dp[j-1] * dp[i-j]
		}
	}
	return dp[n]
}
