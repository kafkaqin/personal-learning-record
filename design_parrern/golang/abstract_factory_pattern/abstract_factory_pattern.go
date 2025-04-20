package main

/**
“同一个工厂能生产一整套风格统一的产品。”
比如你想装修北欧风的房子，就去“北欧风工厂”买沙发、椅子、茶几；如果你要中式风格，就去“中式工厂”。
模式作用
创建一系列相关/配套对象（产品族），如 UI 套件中的按钮+窗口+菜单。
保证产品的一致性（不混搭风格）。
解耦产品的创建过程和使用过程。
*/
import (
	"fmt"
)

type Button interface {
	Render()
}

type Checkbox interface {
	Check()
}

type WindowsButton struct{}

func (wb *WindowsButton) Render() {
	fmt.Println("Render a button in  windows style")
}

type WindowsCheckBox struct{}

func (wb *WindowsCheckBox) Check() {
	fmt.Println("Render a checkbox in  windows style")
}

type MacButton struct{}

func (mb *MacButton) Render() {
	fmt.Println("Render a button in  mac button")
}

type MacCheckBox struct{}

func (mb *MacCheckBox) Check() {
	fmt.Println("Render a checkbox in  mac button")
}

type GUIFactory interface {
	CreateButton() Button
	CreateCheckbox() Checkbox
}

type WindowsFactory struct{}

func (wf *WindowsFactory) CreateCheckbox() Checkbox {
	return &WindowsCheckBox{}
}
func (wf *WindowsFactory) CreateButton() Button {
	return &WindowsButton{}
}

type MacFactory struct{}

func (mf *MacFactory) CreateCheckbox() Checkbox {
	return &MacCheckBox{}
}
func (mf *MacFactory) CreateButton() Button {
	return &MacButton{}
}

type Application struct {
	button   Button
	checkbox Checkbox
}

func NewApplication(factory GUIFactory) *Application {
	return &Application{factory.CreateButton(), factory.CreateCheckbox()}
}

func (app *Application) RenderUI() {
	app.button.Render()
	app.checkbox.Check()
}

func main() {
	var factory GUIFactory
	os := "mac"
	if os == "windows" {
		factory = &WindowsFactory{}
	} else if os == "mac" {
		factory = &MacFactory{}
	}

	app := NewApplication(factory)
	//app.button.Render()
	//app.checkbox.Check()
	app.RenderUI()
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
