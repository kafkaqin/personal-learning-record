package main

import "fmt"

//1.指针逃逸 返回指针

type Demo struct {
	name string
}

func (demo *Demo) setName(name string) {
	demo.name = name
}

func createDemo() *Demo {
	demo := new(Demo)
	return demo
}

// 2.interface{}类型逃逸
func main() {
	demo := createDemo()
	fmt.Println(demo) //demo会逃逸
}

//3.栈空间不足 也会发生逃逸
//4.闭包函数

func Increase() func() int {
	n := 0
	return func() int { //也会逃逸
		n++
		return n
	}
}
