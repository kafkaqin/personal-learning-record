package tree

func revertThreeV1(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}

	root.Left, root.Right = root.Right, root.Left //前序遍历
	if root.Left != nil {
		revertThreeV1(root.Left)
	}
	if root.Right != nil {
		revertThreeV1(root.Right)
	}
	return root
}

func revertThreeV2(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}

	if root.Left != nil {
		revertThreeV1(root.Left)
	}
	if root.Right != nil {
		revertThreeV1(root.Right)
	}
	root.Left, root.Right = root.Right, root.Left //后续遍历
	return root
}

func revertThreeV3(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}

	if root.Left != nil {
		revertThreeV1(root.Left)
	}
	root.Left, root.Right = root.Right, root.Left //中序遍历
	if root.Right != nil {
		revertThreeV1(root.Left)
	}
	return root
}
