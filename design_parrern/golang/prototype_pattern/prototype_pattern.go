package main

import (
	"fmt"
)

type Prototype interface {
	Clone() Prototype
}
type CreatePrototype struct {
	Name string
	Data map[string]string
}

func (c *CreatePrototype) Clone() Prototype {
	newData := make(map[string]string)
	for k, v := range c.Data {
		newData[k] = v
	}
	return &CreatePrototype{c.Name, newData}
}

func main() {
	prototype := &CreatePrototype{
		Name: "Original",
		Data: map[string]string{"key": "value"},
	}
	clone := prototype.Clone().(*CreatePrototype)
	clone.Data["key2"] = "value2"
	clone.Name = "clone"
	fmt.Println("Prototype:", prototype)
	fmt.Println("Clone:", clone)
}
