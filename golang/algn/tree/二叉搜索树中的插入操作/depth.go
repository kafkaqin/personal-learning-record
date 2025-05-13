package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
func insert(cur *TreeNode, val int) *TreeNode {

	if cur == nil {
		cur = &TreeNode{Val: val}
		return cur
	}
	if cur.Val > val { // 在左子节点
		cur.Left = insert(cur.Left, val)
	}

	if cur.Val < val { // 在右子节点
		cur.Right = insert(cur.Right, val)
	}
	return cur
}

// 迭代法
func searchV2(root *TreeNode, val int) *TreeNode {
	newNode := &TreeNode{Val: val}
	if root == nil {
		return newNode
	}

	cur := root
	for {
		if cur.Val > val {
			if cur.Left == nil {
				cur.Left = newNode
				break
			} else {
				cur = cur.Left
			}
		} else {
			if cur.Right == nil {
				cur.Right = newNode
				break
			} else {
				cur = cur.Right
			}
		}
	}
	return root
}
