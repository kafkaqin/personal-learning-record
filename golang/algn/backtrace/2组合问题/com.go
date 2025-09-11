package main

var result [][]int
var path []int

func backtracing(n, k int, startIndex int) {
	if len(path) == k {
		tmp := make([]int, len(path))
		copy(tmp, path)
		result = append(result, tmp)
		return
	}
	for i := startIndex; i <= n; i++ {
		path = append(path, i)
		backtracing(n, k, i+1)
		path = path[:len(path)-1] //回溯
	}
}
