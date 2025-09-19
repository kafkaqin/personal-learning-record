package main

var path []string
var result [][]string

// 排序 sum
func backtracing(s string, startIndex int) {
	if len(s) <= startIndex {
		tmp := make([]string, len(path))
		copy(tmp, path)
		result = append(result, tmp)
		return
	}

	for i := startIndex; i < len(s); i++ {
		if isHuiwen(s[startIndex:i]) {
			path = append(path, string(s[i]))
			backtracing(s, i+1)
			path = path[:len(path)-1]
		} else {
			continue
		}
	}
}

func isHuiwen(s string) bool {
	left := 0
	right := len(s) - 1
	for left < right {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}
	return true
}
