package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// [] 左闭右闭 区间
func traversal(nums []int, left, right int) *TreeNode {
	if left > right {
		return nil
	}
	mid := (right - left) / 2
	root := &TreeNode{Val: nums[mid]}
	root.Left = traversal(nums, left, mid-1)
	root.Right = traversal(nums, mid+1, right)
	return root
}
