package main

// 右边的小孩比左边的小孩得分高的情况
// 左大于右
func maxSum(candy []int, rating []int) int {
	for i := 0; i < len(candy); i++ {
		candy[i] = 1 //糖果 每个人至少得一颗
	}
	result := 0
	size := len(rating)
	for i := 1; i <= size-1; i++ { //右边的小孩比左边的小孩得分高的情况
		if rating[i] > candy[i-1] {
			candy[i] = candy[i-1] + 1
		}
	}
	for i := size - 2; i >= 0; i-- { //左大于右
		if rating[i] > rating[i+1] {
			candy[i] = max(candy[i], candy[i+1]+1)
		}
	}

	for i := 1; i < size; i++ {
		result += candy[i]
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
