package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func getTreeNodeCount(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftCount := getTreeNodeCount(root.Left)
	rightCount := getTreeNodeCount(root.Right)

	return leftCount + rightCount + 1
	// return getTreeNodeCount(root.Left)+getTreeNodeCount(root.Right)+1
}

// 往左遍历和往右遍历的深度是一样的
// 满二叉树: 从左到右依次 顺序地左右叶子节点 2的深度次方-1

func getNumber(root *TreeNode) int {
	if root == nil {
		return 0
	}

	left := root.Left
	right := root.Right
	var leftHeight, rightHeight int
	for left != nil {
		left = left.Left
		leftHeight++
	}
	for right != nil {
		right = right.Right
		rightHeight++
	}
	if leftHeight == rightHeight {
		return 2<<leftHeight - 1 //节点的数量: 2的深度次方-1
	}

	leftNumber := getNumber(left.Left)
	rightNumber := getNumber(right.Right)
	return leftNumber + rightNumber + 1
}
