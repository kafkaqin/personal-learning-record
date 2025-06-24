package main

import (
	"fmt"
	"strconv"
)

func count(num int) int {
	str := []byte(fmt.Sprintf("%d", num))
	size := len(str)
	flag := size

	for i := size - 1; i > 0; i-- {
		if str[i-1] > str[i] {
			str[i-1]--
			flag = i
		}
	}

	for i := flag; i < size; i++ {
		str[i] = '9'
	}

	res, _ := strconv.Atoi(string(str))
	return res
}

func main() {
	fmt.Println(count(1234)) // 输出：1234
	fmt.Println(count(332))  // 输出：299
	fmt.Println(count(120))  // 输出：119
	fmt.Println(count(0))    // 输出：0
}
