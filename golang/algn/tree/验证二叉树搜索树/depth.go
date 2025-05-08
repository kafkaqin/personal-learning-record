package main

import "math"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
// 中序遍历 左中右
func isValueV2(root *TreeNode, array []int) {
	if root == nil {
		return
	}
	isValueV2(root.Left, array)
	array = append(array, root.Val) //验证是否是有序的
	isValueV2(root.Right, array)
}

var maxValue = math.MinInt64

func isValue(root *TreeNode) bool {

	if root == nil {
		return true
	}
	resultLeft := isValue(root.Left)
	if root.Val > maxValue {
		maxValue = root.Val
	} else {
		return false
	}
	resultRight := isValue(root.Right)
	return resultLeft && resultRight
}

var pre *TreeNode

func isValueV1(root *TreeNode) bool {

	if root == nil {
		return true
	}
	resultLeft := isValueV1(root.Left)
	if pre != nil && root.Val <= pre.Val {
		return false
	}
	pre = root
	resultRight := isValueV1(root.Right)
	return resultLeft && resultRight
}
