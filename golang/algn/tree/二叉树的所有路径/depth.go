package main

import "strconv"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//前序遍历

func traversal(root *TreeNode, path []int, result []string) {
	path = append(path, root.Val)
	if root.Right == nil && root.Left == nil {
		tmp := make([]int, len(path))
		copy(tmp, path)
		tmpStr := ""
		for _, i := range tmp {
			tmpStr += "-->" + strconv.Itoa(i) + "<--"
		}
		result = append(result, tmpStr)
	}

	if root.Left != nil {
		traversal(root.Left, path, result)
		path = path[:len(path)-1] //回溯
	}
	if root.Right != nil {
		traversal(root.Right, path, result)
		path = path[:len(path)-1]
	}
}
