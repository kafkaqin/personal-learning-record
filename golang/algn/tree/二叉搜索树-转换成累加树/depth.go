package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 右中左
var pre *TreeNode

func traversal(cur *TreeNode) {
	if cur == nil {
		return
	}
	traversal(cur.Right)
	cur.Val = pre.Val + cur.Val
	pre.Val = cur.Val
	traversal(cur.Left)
}
