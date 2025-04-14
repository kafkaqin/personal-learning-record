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
