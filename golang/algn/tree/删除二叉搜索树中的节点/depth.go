package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 1.没有找到要删除的节点
// 2.叶子节点
// 3.左不空右空
// 4.左空右不空
// 5.左不空右不空
// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
func del(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return root
	}
	if root.Val == key { // 找到删除节点
		if root.Left == nil && root.Right == nil { //叶子节点
			return nil
		} else if root.Left != nil && root.Right == nil {
			return root.Left
		} else if root.Left == nil && root.Right != nil {
			return root.Right
		} else {
			cur := root.Right
			for cur.Left != nil {
				cur = cur.Left
			}
			cur.Left = root.Left
			return root.Right
		}
	}
	if root.Val < key {
		root.Left = del(root.Left, key)
	}

	if root.Val > key {
		root.Right = del(root.Right, key)
	}
	return root
}
