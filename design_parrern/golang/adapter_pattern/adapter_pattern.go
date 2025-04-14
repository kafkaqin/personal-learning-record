package main

/**
适配器模式（Adapter Pattern）是一种结构型设计模式，它的主要作用是在不修改原有代码的情况下，使不兼容的接口能够协同工作。
适配器模式通常用于接口转换，以适配不同的系统、库或 API，从而实现解耦和复用。

对象适配器（Object Adapter）：使用**组合（Composition）**来适配接口。
类适配器（Class Adapter）：使用**继承（Inheritance）**来适配接口。（Golang 不支持类继承，因此一般采用对象适配器）
*/
import "fmt"

type Target interface {
	Request() string
}

type Adaptee struct {
}

func (a *Adaptee) SpecificRequest() string {
	return "Adaptee:Specific Request"
}

type Adapter struct {
	Adaptee *Adaptee
}

func (a *Adapter) Request() string {
	return "Adapter: " + a.Adaptee.SpecificRequest()
}

func main() {

	var target Target

	adaptee := &Adaptee{}

	target = &Adapter{
		Adaptee: adaptee,
	}
	fmt.Println(target.Request())

	vga := &VGA{}

	adapter := &VGAToHDMIAdapter{
		vgaDevice: vga,
	}
	adapter.Display()
}

type HDMI interface {
	Display()
}

type VGA struct {
}

func (v *VGA) ShowImage() {}

type VGAToHDMIAdapter struct {
	vgaDevice *VGA
}

func (v *VGAToHDMIAdapter) Display() {
	fmt.Println("Converting VGA signal to  HDMI....")
	v.vgaDevice.ShowImage()
}
