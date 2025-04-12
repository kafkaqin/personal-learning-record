package main

import "fmt"

/*
*
建造者模式（Builder Pattern）是一种创建型设计模式，
主要用于将一个复杂对象的构建过程与其表示分离，使得同样的构建过程可以创建不同的表示。
通过这种方式，可以将复杂对象的创建步骤封装起来，
从而使客户端代码与对象创建的细节解耦，提高代码的可维护性和扩展性。
*/
type House struct {
	Foundation string
	Walls      string
	Roof       string
}

func (h *House) Show() {
	fmt.Println("House:", h.Foundation, h.Walls, h.Roof)
}

type HourBuilder interface {
	BuildFoundation()
	BuildWalls()
	BuildRoof()
	GetHouse() *House
}
type ConcreteHouseBuilder struct {
	house *House
}

func (c *ConcreteHouseBuilder) BuildFoundation() {
	c.house.Foundation = "Concrete Foundation"
}

func (c *ConcreteHouseBuilder) BuildWalls() {
	c.house.Walls = "Concrete Walls"
}

func (c *ConcreteHouseBuilder) BuildRoof() {
	c.house.Roof = "Concrete Roof"
}

func (c *ConcreteHouseBuilder) GetHouse() *House {
	return c.house
}

func NewConcreteHouseBuilder() *ConcreteHouseBuilder {
	return &ConcreteHouseBuilder{house: &House{}}
}

type HouseDirector struct {
	builder HourBuilder
}

func NewHouseDirector(b HourBuilder) *HouseDirector {
	return &HouseDirector{builder: b}
}

func (h *HouseDirector) ConstructHouse() {
	h.builder.BuildFoundation()
	h.builder.BuildWalls()
	h.builder.BuildRoof()
}
func main() {
	builder := NewConcreteHouseBuilder()
	director := NewHouseDirector(builder)
	director.ConstructHouse()

	house := builder.GetHouse()
	house.Show()
}
