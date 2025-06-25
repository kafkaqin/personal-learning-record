package main

// 每隔两个空节点 就放一个摄像头
// 后续遍历(左右中)
// 下面 三个状态
// 0 无覆盖
// 1 有摄像头的状态
// 2 有覆盖
// 节点为null时 代表 有覆盖的状态
// 左右都是有覆盖
// 左右至少有一个无覆盖
// 左右孩子至少有一个有摄像头
// 如果根节点是无覆盖的状态 需要加上一个摄像头
var result int = 0

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func traver(root *TreeNode) int {
	if root == nil { // 左右都是有覆盖
		return 2 // 2 有覆盖
	}
	left := traver(root.Left)
	right := traver(root.Right)
	if left == 2 && right == 2 {
		return 0 // 无覆盖
	}
	if left == 0 || right == 0 { // 左右至少有一个无覆盖
		result++
		return 1 //有摄像头的状态
	}

	if left == 1 || right == 1 { // 左右孩子至少有一个有摄像头
		return 2
	}
	return -1
}

func main() {
	tree := &TreeNode{}
	root := traver(tree)
	if root == 0 {
		result += 1
	}
}
