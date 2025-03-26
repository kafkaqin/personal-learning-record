package main

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

}
