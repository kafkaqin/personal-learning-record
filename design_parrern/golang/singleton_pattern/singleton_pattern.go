package main

import (
	"fmt"
	"sync"
)

/**
单例设计模式（Singleton Pattern）是一种创建型设计模式，它确保一个类只有一个实例，并提供一个全局访问点。
该模式常用于需要集中管理资源或共享数据的场景，例如配置管理、日志记录、连接池等。
*/
/**
主要特点
唯一性：单例模式确保一个类只有一个实例，防止多个实例同时存在。

全局访问：提供一个全局访问点，使得程序中任何地方都可以方便地获取这个唯一实例。

延迟初始化：通常采用延迟初始化技术，只有在需要时才创建实例，同时需要保证线程安全。
*/

type Singleton struct {
	Data int
}

var instance *Singleton
var once sync.Once

func GetInstance() *Singleton {
	once.Do(func() {
		instance = &Singleton{Data: 42}
	})
	return instance
}

func main() {
	singleton1 := GetInstance()
	singleton2 := GetInstance()
	fmt.Println("singleton1", singleton1.Data)
	fmt.Println("singleton2", singleton2.Data)
}
