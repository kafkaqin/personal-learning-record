## 二叉树的种类:
1.满二叉树
节点的数量 2的k次方-1

2.完全二叉树

3.二叉搜索树

4.平衡二叉搜索树
左子树和右子树的高度差小于或者等于1

## 二叉树存储方式
1.链式存储
2.线性存储 2xi+1(左孩子) 2xi+2(右孩子)

## 遍历方式
1.深度优先搜索(前中后序遍历) --- 递归实现 迭代法  栈
2.广度优先搜索(层序遍历) --- 迭代法 队列

## 二叉树的定义
```golang
type Node struct {
	Left *Node
	Right *Node 
	Value int 
}
```