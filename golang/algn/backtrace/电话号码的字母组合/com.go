package main

var s string
var result []string
var letteryMap = map[string]string{
	"0": `""`,
	"1": `''`,
	"2": "abc",
	"3": "def",
	"4": "ghi",
	"5": "jkl",
	"6": "mno",
	"7": "pqrs",
	"8": "tuv",
	"9": "wxyz",
}

func backtracing(digits string, index int) {
	if len(digits) == index {
		result = append(result, s)
		return
	}

	//dig := digits[index] - '0'
	dig := digits[index]
	letteryx := letteryMap[string(dig)]
	for i := 0; i < len(letteryx); i++ {
		s = s + string(letteryx[i])
		backtracing(digits, i+1)
		s = s[:len(s)-1] //回溯
	}
}
