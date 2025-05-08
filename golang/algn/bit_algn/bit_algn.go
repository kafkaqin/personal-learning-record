package main

import "fmt"

func main() {
	a := 60
	b := 13
	fmt.Printf("a = %d %08b, b = %d %08b\n", a, a, b, b)
	and := a & b
	fmt.Printf("and = %d %08b\n", and, and)

	or := a | b
	fmt.Printf("or = %d %08b\n", or, or)
	xor := a ^ b
	fmt.Printf("xor = %d %08b\n", xor, xor)

	notA := ^a
	fmt.Printf("notA = %d %08b\n", notA, notA)

	leftShift := a << 1
	fmt.Printf("leftShift = %d %08b\n", leftShift, leftShift)
	rightShift := a >> 3
	fmt.Printf("rightShift = %d %08b\n", rightShift, rightShift)
}
