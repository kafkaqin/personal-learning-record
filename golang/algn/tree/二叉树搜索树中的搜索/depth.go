package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
func search(root *TreeNode, val int) *TreeNode {
	if root == nil || root.Val == val {
		return root
	}
	result := &TreeNode{}
	if val < root.Val {
		result = search(root.Left, val)
	}
	if val > root.Val {
		result = search(root.Right, val)
	}
	return result
}

func searchV2(root *TreeNode, val int) *TreeNode {
	for root != nil {
		if root.Val < val {
			root = root.Left
		} else if val > root.Val {
			root = root.Right
		} else {
			return root
		}
	}
	return nil
}
