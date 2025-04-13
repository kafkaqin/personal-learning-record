package main

/**
工厂模式是一种创建型设计模式，它的核心思想是将对象的创建过程封装到专门的工厂中，使得客户端无需直接依赖具体的产品类，从而降低了系统的耦合性并提高了扩展性。工厂模式主要包括以下几种变体：

简单工厂模式：由一个工厂类根据参数决定创建哪种产品实例。虽然简单易懂，但违反了“开闭原则”，因为每增加一种产品都需要修改工厂类。

工厂方法模式：通过定义一个抽象工厂接口，由子类决定具体产品的实例化。这样新增产品时，只需增加相应的工厂子类，不需要修改已有代码。

抽象工厂模式：在工厂方法模式的基础上进一步扩展，用于创建一系列相关或相互依赖的产品族。
*/
import (
	"fmt"
)

type Button interface {
	Paint()
}

type CheckBox interface {
	Paint()
}

type WindowsButton struct{}

func (wb *WindowsButton) Paint() {
	fmt.Println("Render a button in  windows style")
}

type WindowsCheckBox struct{}

func (wb *WindowsCheckBox) Paint() {
	fmt.Println("Render a checkbox in  windows style")
}

type MacButton struct{}

func (mb *MacButton) Paint() {
	fmt.Println("Render a button in  mac button")
}

type MacCheckBox struct{}

func (mb *MacCheckBox) Paint() {
	fmt.Println("Render a checkbox in  mac button")
}

type GUIFactory interface {
	CreateButton() Button
	CreateCheckBox() CheckBox
}

type WindowsFactory struct{}

func (wf *WindowsFactory) CreateCheckBox() CheckBox {
	return &WindowsCheckBox{}
}
func (wf *WindowsFactory) CreateButton() Button {
	return &WindowsButton{}
}

type MacFactory struct{}

func (mf *MacFactory) CreateCheckBox() CheckBox {
	return &MacCheckBox{}
}
func (mf *MacFactory) CreateButton() Button {
	return &MacButton{}
}

func RenderUI(factory GUIFactory) {
	button := factory.CreateButton()
	checkbox := factory.CreateCheckBox()
	button.Paint()
	checkbox.Paint()
}

func main() {
	var factory GUIFactory
	os := "mac"
	if os == "windows" {
		factory = &WindowsFactory{}
	} else if os == "mac" {
		factory = &MacFactory{}
	}

	RenderUI(factory)

	var factory2 ShapeFactory
	factory2 = &CircleFactory{}
	circle := factory2.CreateShape()
	circle.Draw()

	factory2 = &RectangleFactory{}
	rectangle := factory2.CreateShape()
	rectangle.Draw()
}

// -=====new
type Shape interface {
	Draw()
}

type Circle struct{}

func (c *Circle) Draw() {
	fmt.Println("Drawing a Circle")
}

type Rectangle struct {
}

func (r *Rectangle) Draw() {
	fmt.Println("Drawing a Rectangle")
}

type ShapeFactory interface {
	CreateShape() Shape
}

type CircleFactory struct{}

func (c *CircleFactory) CreateShape() Shape {
	return &Circle{}
}

type RectangleFactory struct{}

func (r *RectangleFactory) CreateShape() Shape {
	return &Rectangle{}
}
