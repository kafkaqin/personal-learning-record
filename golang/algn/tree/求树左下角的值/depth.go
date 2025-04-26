package main

import "math"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 深度 前中后序都可以 只要保证左节点在右节点后面就OK
var maxDepth = math.MinInt64
var result int

func traversal(root *TreeNode, depth int) {
	if root.Left == nil && root.Right == nil {
		if depth >= maxDepth {
			maxDepth = depth
			result = root.Val
		}
	}

	if root.Left != nil {
		depth++
		traversal(root.Left, depth)
		depth--
	}

	if root.Right != nil {
		depth++
		traversal(root.Right, depth)
		depth--
	}
}
