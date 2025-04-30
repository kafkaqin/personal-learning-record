package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//平衡二叉树 任意一个节点左右子树的高度差小于等于1
// 高度 距离叶子节点距离 后续
// 一起操作两个二叉树

func mergeTree(tree1 *TreeNode, tree2 *TreeNode) *TreeNode {
	if tree1 == nil {
		return tree2
	}
	if tree2 == nil {
		return tree1
	}

	tree1.Val = tree1.Val + tree2.Val

	tree1.Left = mergeTree(tree1.Left, tree2.Left)
	tree1.Right = mergeTree(tree1.Right, tree2.Right)

	return tree1
}
