package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func traversal(inOrder []int, postOrder []int) *TreeNode {
	if len(postOrder) == 0 { //后序长度为0
		return nil
	}
	rootValue := postOrder[len(postOrder)-1]
	root := &TreeNode{Val: rootValue}
	if len(postOrder) == 1 {
		return root
	}

	i := 0
	for index := 0; index < len(inOrder); index++ { //切中序数组
		if inOrder[index] == rootValue { // 获取
			i = index
			break
		}
	}

	LeftInOrder, rightInOrder := inOrder[:i], inOrder[i+1:]
	j := 0
	for index := 0; index < len(postOrder); index++ { //切中序数组
		if postOrder[index] == inOrder[i-1] { // 获取
			j = index
			break
		}
	}
	LeftPostOrder, rightPostOrder := postOrder[:j], postOrder[j+1:]

	root.Left = traversal(LeftInOrder, LeftPostOrder)
	root.Right = traversal(rightInOrder, rightPostOrder)
	return root
}
