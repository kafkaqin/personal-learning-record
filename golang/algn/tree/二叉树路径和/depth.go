package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// count 目标值
func traversal(root *TreeNode, count int) bool {
	if root.Left == nil && root.Right == nil {
		if count == 0 {
			return true
		} else {
			return false
		}
	}

	if root.Left != nil {
		count = count - root.Left.Val
		if traversal(root.Left, count) {
			return true
		}
		count = count + root.Val

	}

	if root.Right != nil {
		count = count - root.Right.Val
		if traversal(root.Right, count) {
			return true
		}
		count += root.Val
	}
	return false
}
