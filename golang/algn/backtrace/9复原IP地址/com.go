package main

var result []string

// 排序 sum
func backtracing(s string, startIndex int, pointSum int) {
	if pointSum == 3 {
		if isValid(s, startIndex, len(s)-1) {
			result = append(result, s)
			return
		}

	}

	for i := startIndex; i < len(s); i++ {
		if isValid(s, startIndex, i) {
			s = s + "."
			pointSum += 1
			backtracing(s, i+2, pointSum)
			pointSum -= 1
			s = s[:len(s)-1]
		}
	}
}

func isValid(s string, left int, right int) bool { //[]左闭右闭区间
	//不能以0开头
	// 不能大于255
	//不能包含非法字符串

	if s[left] == '0' {
		return false
	}
	if len(s[left:right]) > 3 {
		return false
	}
	//0-9合法,其他非法
	return true
}
