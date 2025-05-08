package main

import "math"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var pre *TreeNode
var result = math.MaxInt64

// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
func search(cur *TreeNode) {
	if cur == nil {
		return
	}
	search(cur.Left)
	if pre != nil {
		curVal := cur.Val - pre.Val
		if curVal < result {
			result = curVal
		}
	}
	pre = cur
	search(cur.Right)

}
