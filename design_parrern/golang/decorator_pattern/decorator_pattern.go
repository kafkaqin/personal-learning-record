package main

/**
装饰器模式（Decorator Pattern）是一种结构型设计模式，它的核心思想是在不修改原有对象的情况下，为其动态添加新功能。
装饰器模式比继承更加灵活，可以在运行时组合不同的行为。
*/
import (
	"fmt"
)

type Component interface {
	Operation() string
}

type ConcreteComponent struct{}

func (c *ConcreteComponent) Operation() string {
	return "ConcreteComponent"
}

type Decorator struct {
	Component Component
}

func (d *Decorator) Operation() string {
	// 这里可以做其他的事情
	if d.Component == nil {
		return "Decorator"
	}
	return d.Component.Operation()
}

type DecoratorComponentA struct {
	Decorator
}

func (a *DecoratorComponentA) Operation() string {
	return "AA " + a.Component.Operation() + " AA"
}

type DecoratorComponentB struct {
	Decorator
}

func (b *DecoratorComponentB) Operation() string {
	return "BB" + b.Component.Operation() + "BB"
}

func main() {
	var component Component = &ConcreteComponent{}
	fmt.Println("Base Component: ", component.Operation())
	decoratorComponentA := &DecoratorComponentA{
		Decorator{
			Component: component,
		},
	}
	fmt.Println("After ConcreteDecoratorA: ", decoratorComponentA.Operation())

	decoratorComponentB := &DecoratorComponentB{
		Decorator{
			Component: component,
		},
	}
	fmt.Println("After ConcreteDecoratorB: ", decoratorComponentB.Operation())

	simpleCoffee := &SimpleCoffee{}
	fmt.Println(simpleCoffee.Description(), " Cost:", simpleCoffee.Cost())

	milkCoffee := &MilkDecorator{
		CoffeeDecorator{simpleCoffee},
	}
	fmt.Println(milkCoffee.Description(), " Cost:", milkCoffee.Cost())

	sugarCoffee := &SugarDecorator{
		CoffeeDecorator{milkCoffee},
	}
	fmt.Println(sugarCoffee.Description(), " Cost:", sugarCoffee.Cost())
}

type Coffee interface {
	Cost() int
	Description() string
}

type SimpleCoffee struct{}

func (s *SimpleCoffee) Cost() int {
	return 10
}

func (s *SimpleCoffee) Description() string {
	return "Simple Coffee"
}

type CoffeeDecorator struct {
	coffee Coffee
}

func (c *CoffeeDecorator) Cost() int {
	return c.coffee.Cost()
}

func (c *CoffeeDecorator) Description() string {
	return c.coffee.Description()
}

type MilkDecorator struct {
	CoffeeDecorator
}

func (m *MilkDecorator) Cost() int {
	return m.coffee.Cost() + 2
}

func (m *MilkDecorator) Description() string {
	return m.coffee.Description() + ", with Milk"
}

type SugarDecorator struct {
	CoffeeDecorator
}

func (s *SugarDecorator) Cost() int {
	return s.coffee.Cost() + 1
}
func (s *SugarDecorator) Description() string {
	return s.coffee.Description() + ", with Sugar"
}
