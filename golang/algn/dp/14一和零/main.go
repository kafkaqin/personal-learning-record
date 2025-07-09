package main

// m个0  n个1
//dp[i][j]的含义: 装满i个0 ,j个1最大背 dp[i][j]个物品(dp[m][n])
//纯01背包的递推公式 dp[j] = max(dp[j],dp[j-weight[i]]+values[i])

// dp[i][j] = max(dp[i][j], dp[i-x][j-y]+1)

// dp[0][0]

func bag(m, n int, wupin []string) int {
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	// dp[0][0] = 0 初始化
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dp[i][j] = 0
		}
	}

	// wupin
	for k := 0; k < len(wupin); k++ { //物品
		x, y := 0, 0
		for _, c := range wupin[k] {
			if c == '0' {
				x++
			} else {
				y++
			}
		}

		for i := m; i >= x; i-- { //背包 两个维度
			for j := n; j >= y; j-- {
				dp[i][j] = max(dp[i][j], dp[i-x][j-y]+1)
			}
		}
	}

	return dp[m][n]
}
