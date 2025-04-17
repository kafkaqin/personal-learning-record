package main

import "fmt"

/**
代理设计模式（Proxy Pattern）是一种结构型设计模式，为其他对象提供一个代理以控制对这个对象的访问。
就像你在访问一个资源时，先通过“中间人”——代理对象，由它来决定是否、何时以及如何访问真正的对象。
*/
/**
作用（为什么用代理模式）
控制访问：比如权限控制、延迟加载、远程调用等。
增加额外功能：记录日志、性能监控、缓存等。
隔离复杂逻辑：隐藏目标对象的复杂实现。
*/

type Image interface {
	Display()
}

type RealImage struct {
	filename string
}

func (r *RealImage) Display() {
	fmt.Println("Display RealImage:", r.filename)
}

func NewRealImage(filename string) *RealImage {
	fmt.Println("NewRealImage:", filename)
	return &RealImage{filename}
}

type ProxyImage struct {
	filename  string
	realImage *RealImage
}

func (p *ProxyImage) Display() {
	if p.realImage == nil {
		p.realImage = NewRealImage(p.filename)
	}
	p.realImage.Display()
}

func main() {
	image := &ProxyImage{
		filename: "test.png",
		//realImage: NewRealImage("test.png"),
	}
	fmt.Println("第一次访问图片Display()")
	image.Display()

	fmt.Println("第二次访问图片Display()")
	image.Display()
}
