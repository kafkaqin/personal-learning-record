package main

// 1. 按身高排序
// 2. 按 k 排序
func maxSum(people [][]int) [][]int {
	for i := 0; i < len(people); i++ {
		for j := i + 1; j < len(people); j++ {
			if people[i][0] > people[j][0] {
				tmp := people[i]
				people[i] = people[j]
				people[j] = tmp
			}
		}
	}

	result := make([][]int, len(people))
	for i := 0; i < len(people); i++ {
		pos := people[i][1]
		result[pos] = people[i]
	}
	return people
}
