package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
func traversal(root *TreeNode, low, high int) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val < low { // 在左子节点
		right := traversal(root.Right, low, high)
		return right
	}

	if root.Val > high {
		left := traversal(root.Left, low, high)
		return left
	}

	root.Left = traversal(root.Left, low, high)
	root.Right = traversal(root.Right, low, high)
	return root
}
