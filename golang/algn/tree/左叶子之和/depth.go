package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 后续遍历 左叶子之和: 所有左叶子节点的孩子的值之和
func traversal(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil { //叶子节点
		return 0
	}
	leftNumber := traversal(root.Left)
	if root.Left.Left == nil && root.Left.Right == nil {
		leftNumber = root.Left.Val
	}
	rightNumber := traversal(root.Right)
	sum := leftNumber + rightNumber
	return sum
}
