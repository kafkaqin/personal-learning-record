package main

import "fmt"

type Component interface {
	Display(depth int)
}

type Employee struct {
	Name string
}

func (e *Employee) Display(depth int) {
	for i := 0; i < depth; i++ {
		fmt.Printf("-")
	}
	fmt.Printf("Employee %s is displaying at depth %d\n", e.Name, depth)
}

type Department struct {
	Name     string
	Children []Component
}

func (d *Department) Add(child Component) {
	d.Children = append(d.Children, child)
}

func (d *Department) Display(depth int) {
	for i := 0; i < depth; i++ {
		fmt.Printf("-")
	}
	fmt.Println(d.Name)
	for _, child := range d.Children {
		child.Display(depth + 1)
	}
}

func main() {
	emp1 := new(Employee)
	emp1.Name = "John"

	emp2 := new(Employee)
	emp1.Name = "Jane"

	emp3 := new(Employee)
	emp1.Name = "Alice"

	emp4 := new(Employee)
	emp1.Name = "Bob"

	depart1 := new(Department)
	depart1.Name = "Engineering"
	depart1.Add(emp1)
	depart1.Add(emp2)
	depart2 := new(Department)
	depart2.Name = "HR"

	depart2.Add(emp3)
	depart2.Add(emp4)

	company := &Department{
		Name: "Company",
	}

	company.Add(depart1)
	company.Add(depart2)

	company.Display(0)
}
