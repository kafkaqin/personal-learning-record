package main

import "fmt"

/**
享元模式（Flyweight Pattern）是一种结构型设计模式，通过共享大量细粒度对象来节省内存。
当你需要创建成千上万个“看起来很像”的对象时，享元模式可以帮助你复用已有实例，避免重复创建，
极大地降低内存使用。
*/
/**
作用（为什么用享元模式）
节省内存（尤其适用于海量对象）
提高性能
避免重复构建相同对象
*/

type ChessPiece interface {
	Display(x, y int)
}

type blackPiece struct{}

func (b *blackPiece) Display(x, y int) {
	fmt.Println("black piece", x, y)
}

type whitePiece struct{}

func (w *whitePiece) Display(x, y int) {
	fmt.Println("White piece", x, y)
}

type PieceFactory struct {
	blackPiece ChessPiece
	whitePiece ChessPiece
}

func NewPieceFactory() *PieceFactory {
	return &PieceFactory{
		blackPiece: &blackPiece{},
		whitePiece: &whitePiece{},
	}
}

func (f *PieceFactory) GetPiece(color string) ChessPiece {
	if color == "black" {
		return f.blackPiece
	}
	return f.whitePiece
}

func main() {
	factory := NewPieceFactory()
	p1 := factory.GetPiece("black")
	p2 := factory.GetPiece("white")
	p3 := factory.GetPiece("black")

	p1.Display(1, 1)
	p2.Display(2, 2)
	p3.Display(3, 3)
}
