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
	if cur.Val > p || cur.Val > q { // 在左子节点
		leftNode := search(cur.Left, p, q)
		if leftNode != nil {
			return leftNode
		}
	}

	if cur.Val < p || cur.Val < q { // 在右子节点
		rightNode := search(cur.Right, p, q)
		if rightNode != nil {
			return rightNode
		}
	}
	return cur
}

// 迭代法
func searchV2(cur *TreeNode, p int, q int) *TreeNode {
	for cur != nil {
		if cur.Val > p && cur.Val > q {
			cur = cur.Left
		} else if cur.Val < p && cur.Val < q {
			cur = cur.Right
		} else {
			return cur
		}
	}
	return nil
}
