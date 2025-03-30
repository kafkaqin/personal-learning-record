package main

import "fmt"

type Shape interface {
	Accept(visitor ShapeVisitor)
}

type Circle struct {
	radius float64
}

func (c *Circle) Accept(visitor ShapeVisitor) {
	visitor.VisitCircle(c)
}

type Rectangle struct {
	width, height float64
}

func (r *Rectangle) Accept(visitor ShapeVisitor) {
	visitor.VisitRectangle(r)
}

type ShapeVisitor interface {
	VisitCircle(c *Circle)
	VisitRectangle(r *Rectangle)
}

type AreaVisitor struct{}

func (a *AreaVisitor) VisitCircle(c *Circle) {
	area := 3.14 * c.radius * c.radius
	fmt.Println("Circle Area is:", area)
}

func (a *AreaVisitor) VisitRectangle(r *Rectangle) {
	area := r.width * r.height
	fmt.Println("Rectangle Area is:", area)
}

type PerimeterVisitor struct{}

func (p *PerimeterVisitor) VisitCircle(c *Circle) {
	perimeter := 2 * 3.14 * c.radius
	fmt.Println("Circle Perimeter is:", perimeter)
}

func (p *PerimeterVisitor) VisitRectangle(r *Rectangle) {
	perimeter := 2 * (r.width + r.height)
	fmt.Println("Rectangle Perimeter is:", perimeter)
}

func main() {
	shapes := []Shape{
		&Circle{5},
		&Rectangle{
			width:  3,
			height: 4,
		},
	}
	areaVisitor := &AreaVisitor{}
	for _, shape := range shapes {
		shape.Accept(areaVisitor)
	}
	perimeterVisitor := &PerimeterVisitor{}
	for _, shape := range shapes {
		shape.Accept(perimeterVisitor)
	}
}
