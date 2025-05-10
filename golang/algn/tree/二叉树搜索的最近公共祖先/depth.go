package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

var pre *TreeNode
var count = make(map[int]int)
var maxCount int
var result = make([]int, 0)

// 后续遍历
// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
func search(cur *TreeNode, p int, q int) *TreeNode {
	if cur == nil {
		return cur
	}
	if cur.Val == p || cur.Val == q {
		return cur
	}
	leftNode := search(cur.Left, p, q)
	rightNode := search(cur.Right, p, q)
	if leftNode != nil && rightNode != nil {
		return cur
	}
	if leftNode == nil && rightNode != nil {
		return rightNode
	}

	if leftNode != nil {
		return leftNode
	}
	return nil
}
