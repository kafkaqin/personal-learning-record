package main

import "fmt"

/**
解释器模式（Interpreter Pattern）是一种行为型设计模式，
它定义了一种语言的文法，并利用该文法来解释语言中的句子。
通过将语言中的每个符号或规则表示为一个类，
解释器模式能够将复杂的语言解析问题拆分成一系列简单的解释任务。
这种模式特别适用于实现领域特定语言（DSL）或简单的脚本语言。
*/

type Expression interface {
	Interpret() int
}

type NumberExpression struct {
	Value int
}

func (n *NumberExpression) Interpret() int {
	return n.Value
}

type PlusExpression struct {
	Left, Right Expression
}

func (p *PlusExpression) Interpret() int {
	return p.Left.Interpret() + p.Right.Interpret()
}

type MinusExpression struct {
	Left, Right Expression
}

func (m *MinusExpression) Interpret() int {
	return m.Left.Interpret() - m.Right.Interpret()
}

func main() {
	expression := &MinusExpression{
		Left: &PlusExpression{
			Left:  &NumberExpression{Value: 7},
			Right: &NumberExpression{Value: 8},
		},
		Right: &NumberExpression{Value: 3},
	}

	result := expression.Interpret()
	fmt.Printf("表达式的结果为: %d\n", result)
}
