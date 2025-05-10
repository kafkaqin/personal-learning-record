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

// 什么是二叉搜索树: 根节点要比左子树大，比右子树小
func search(cur *TreeNode) {
	if cur == nil {
		return
	}
	search(cur.Left)
	if pre == nil {
		count[cur.Val] = 1
	}
	if pre != nil && cur.Val == pre.Val {
		count[cur.Val]++
	}
	if count[cur.Val] == maxCount {
		result = append(result, cur.Val)
	}

	if count[cur.Val] > maxCount {
		maxCount = count[cur.Val]
		result = make([]int, 0)
		result = append(result, cur.Val)
	}
	pre = cur
	search(cur.Right)
}
