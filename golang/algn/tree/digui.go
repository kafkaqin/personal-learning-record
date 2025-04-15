package tree

//递归三部曲
//1.确定递归函数的参数和返回值
//2.确定终止条件
//3.确定单层递归的逻辑

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func qianxu(cur *TreeNode, list []int) {
	if cur == nil {
		return
	}
	list = append(list, cur.Val) // 中
	qianxu(cur.Left, list)       // 左
	qianxu(cur.Right, list)      // 右
}

func zhongxu(cur *TreeNode, list []int) {
	if cur == nil {
		return
	}
	zhongxu(cur.Left, list)      // 左
	list = append(list, cur.Val) // 中
	zhongxu(cur.Right, list)     // 右
}

func houxu(cur *TreeNode, list []int) {
	if cur == nil {
		return
	}
	houxu(cur.Left, list)        // 左
	houxu(cur.Right, list)       // 右
	list = append(list, cur.Val) // 中
}

//下面试迭代法实现前中后序遍历

// qianxu2 前序遍历
func qianxu2(root *TreeNode) []int {
	stack := make([]*TreeNode, 0)
	result := make([]int, 0)
	stack = append(stack, root)
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1] //中
		if cur != nil {
			result = append(result, cur.Val)
		} else {
			continue
		}
		//先放入节点的右孩子 再放左孩子
		stack = append(stack, cur.Right) //右
		stack = append(stack, cur.Left)  // 左
	}
	return result
}

// houxu2 后序遍历 左右中
func houxu2(root *TreeNode) []int {
	stack := make([]*TreeNode, 0)
	result := make([]int, 0)
	stack = append(stack, root)
	for len(stack) > 0 {
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1] //中
		if cur != nil {
			result = append(result, cur.Val)
		} else {
			continue
		}
		//先放入节点的右孩子 再放左孩子
		stack = append(stack, cur.Left)  // 左
		stack = append(stack, cur.Right) //右
	}
	//反转
	res := []int{}
	for i := len(result) - 1; i >= 0; i-- {
		res = append(res, result[i])
	}
	return res
}
