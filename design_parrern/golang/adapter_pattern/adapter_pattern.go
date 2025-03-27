package main

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
}
