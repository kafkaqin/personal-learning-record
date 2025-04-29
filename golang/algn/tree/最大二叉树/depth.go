package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func traversal(nums []int) *TreeNode {
	if len(nums) == 1 { //后序长度为1时 已经到叶子节点
		return &TreeNode{Val: nums[0]}
	}
	maxValue := 0
	j := 0
	for index := 0; index < len(nums); index++ { //切中序数组
		if maxValue < nums[index] { // 获取
			j = index
			maxValue = nums[index]
		}
	}
	root := &TreeNode{Val: maxValue} //中
	if j > 0 {
		root.Left = traversal(nums[0:j]) //左
	}
	if j < len(nums)-1 {
		root.Right = traversal(nums[j+1:]) //右
	}
	return root
}
