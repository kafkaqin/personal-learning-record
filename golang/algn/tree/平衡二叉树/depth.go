package main

import "math"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//平衡二叉树 任意一个节点左右子树的高度差小于等于1
// 高度 距离叶子节点距离 后续
// 深度 距离跟节点的 前序

func getHeight(node *TreeNode) int {
	if node == nil {
		return 0 //高度为0
	}
	//左右中
	leftHeight := getHeight(node.Left)
	if leftHeight == -1 {
		return -1
	}
	rightHeight := getHeight(node.Right)
	if rightHeight == -1 {
		return -1
	}

	result := int(math.Abs(float64(rightHeight - leftHeight)))
	if result > 1 {
		result = -1
	} else {
		result = 1 + max(leftHeight, rightHeight)
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
