package main

var result [][]int
var path []int

func backtracing(targetSum, k int, sum, startIndex int) {
	if len(path) == k {
		if targetSum == sum {
			tmp := make([]int, len(path))
			copy(tmp, path)
			result = append(result, tmp)
		}
		return
	}
	for i := startIndex; i <= 9; i++ {
		sum += i
		path = append(path, i)
		backtracing(targetSum, k, sum, i+1)
		sum -= i
		path = path[:len(path)-1] //回溯
	}
}

func backtracingV2(targetSum, k int, sum, startIndex int) {
	if sum > targetSum { //垂直方向
		return
	}
	if len(path) == k {
		if targetSum == sum {
			tmp := make([]int, len(path))
			copy(tmp, path)
			result = append(result, tmp)
		}
		return
	}
	for i := startIndex; i <= (9 - (k - len(path)) + 1); i++ { //水平方向
		sum += i
		path = append(path, i)
		backtracingV2(targetSum, k, sum, i+1)
		sum -= i
		path = path[:len(path)-1] //回溯
	}
}
