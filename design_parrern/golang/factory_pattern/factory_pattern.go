package main

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
}
