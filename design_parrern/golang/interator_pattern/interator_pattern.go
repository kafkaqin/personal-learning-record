package main

import "fmt"

/**
迭代器模式（Iterator Pattern）是一种行为型设计模式，
它提供了一种方法，可以顺序访问聚合对象（如集合或容器）中的各个元素，而无需暴露该对象的内部表示。
通过迭代器，客户端可以专注于遍历数据，而不需要关心集合内部的存储细节。
*/

type Interator interface {
	HasNext() bool
	Next() interface{}
}

type Aggregate interface {
	CreateInterator() Interator
}

type IntCollection struct {
	Items []int
}

func (i *IntCollection) CreateInterator() Interator {
	return &IntInterator{
		collection: i,
		index:      0,
	}
}

type IntInterator struct {
	collection *IntCollection
	index      int
}

func (i *IntInterator) HasNext() bool {
	return i.index < len(i.collection.Items)
}

func (i *IntInterator) Next() interface{} {
	for i.HasNext() {
		item := i.collection.Items[i.index]
		i.index++
		return item
	}
	return nil
}

func main() {
	collection := &IntCollection{
		Items: []int{1, 2, 3, 4, 5, 6, 7, 87, 8, 9},
	}
	interator := collection.CreateInterator()
	if interator.HasNext() {
		fmt.Println("HasNext", interator.Next())
	}
}
