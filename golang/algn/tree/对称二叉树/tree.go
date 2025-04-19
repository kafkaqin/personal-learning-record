package main

type TreeNode struct {
	Value int
	Left  *TreeNode
	Right *TreeNode
}

// 采用 后续遍历
// 1.参数和返回值
// 2
func compare(left, right *TreeNode) bool {
	if left == nil && right != nil {
		return false
	} else if left != nil && right == nil {
		return false
	} else if left == nil && right == nil {
		return true
	} else {
		if left.Value != right.Value {
			return false
		}
	}
	outsite := compare(left.Left, right.Right) //左 右 中
	insite := compare(left.Right, right.Left)  // 右 左 中   后续遍历

	return outsite && insite
}
