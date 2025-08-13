package main

// dp[0] 不偷当前节点的最大金钱 dp[1] 偷当前节点所获得的最大值
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func rebTree(root *TreeNode) []int {
	dp := make([]int, 2)
	if root == nil {
		return dp
	}
	leftDp := rebTree(root.Left)
	rightDp := rebTree(root.Right)
	dp[1] = root.Val + leftDp[0] + rightDp[0]                       //偷当前节点
	dp[0] = max(leftDp[0], leftDp[1]) + max(rightDp[1], rightDp[0]) // 不偷当前节点
	return dp
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
	var root *TreeNode
	result := rebTree(root)
	max(result[0], result[1])
}
