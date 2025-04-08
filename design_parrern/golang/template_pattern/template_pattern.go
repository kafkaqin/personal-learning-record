package main

import "fmt"

/**
模板方法设计模式（Template Method Pattern）是一种行为型设计模式，它定义了一个算法的骨架，
将算法中的某些步骤延迟到子类中实现。也就是说，模板方法在基类中规定了执行步骤的顺序，
而将具体步骤的实现交由子类完成，从而使得子类可以在不改变整体流程的情况下重新定义某些步骤的实现方式。
*/

type Beverage interface {
	BoilWater()
	Brew()
	PourInCup()
	AddCondiments()
	PrepareRecipe()
}

type BaseBeverage struct{}

// BoilWater 公共步骤：烧水
func (b *BaseBeverage) BoilWater() {
	fmt.Println("Boiling water")
}

// PourInCup 公共步骤：倒入杯中
func (b *BaseBeverage) PourInCup() {
	fmt.Println("Pouring into cup")
}

type Coffee struct {
	BaseBeverage
}

func (c *Coffee) Brew() {
	fmt.Println("Dripping coffee through filter")
}

func (c *Coffee) AddCondiments() {
	fmt.Println("Adding sugar and milk")
}

func (c *Coffee) PrepareRecipe() {
	c.BoilWater()
	c.Brew()
	c.PourInCup()
	c.AddCondiments()
}

type Tea struct {
	BaseBeverage
}

func (t *Tea) AddCondiments() {
	fmt.Println("Adding lemon")
}

func (t *Tea) PrepareRecipe() {
	t.BoilWater()
	t.Brew()
	t.PourInCup()
	t.AddCondiments()
}

func (t *Tea) Brew() {
	fmt.Println("Steeping the tea")
}

func main() {
	fmt.Println("Preparing Coffee:")
	coffee := Coffee{}
	coffee.PrepareRecipe()

	fmt.Println("Preparing Tea:")
	tea := Tea{}
	tea.PrepareRecipe()
}
