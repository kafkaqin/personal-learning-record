package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//深度 任意一个节点到根节点的距离 从上往下 前序遍历 中左右
//高度 从下往上遍历 后序遍历 左右中

func getHeightV1(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return max(getHeight(root.Left), getHeight(root.Right)) + 1
}

func getHeight(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftHeight := getHeight(root.Left)
	rightHeight := getHeight(root.Right)

	height := max(leftHeight, rightHeight) + 1
	return height
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
