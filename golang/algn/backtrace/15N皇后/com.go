package main

var result [][][]int

func backtracing(checkerboard [][]int, n int, row int) {
	if row == n {
		tmp := make([][]int, len(checkerboard))
		copy(tmp, checkerboard)
		result = append(result, tmp)
		return
	}

	for i := 0; i < n; i++ { //列
		if isValid(row, i, checkerboard, n) {
			checkerboard[row][i] = 1
			backtracing(checkerboard, n, row+1)
			checkerboard[row][i] = 0
		}
	}
}

func isValid(row int, i int, checkerboard [][]int, n int) bool {
	return true
}
