package main

import "fmt"

/**
状态设计模式（State Pattern）是一种行为型设计模式，它允许一个对象在其内部状态改变时改变它的行为，
看起来就像改变了它的类一样。
 作用（为什么用状态模式）
将状态相关的行为封装到独立的状态类中，使状态切换更清晰。
避免在一个类中写大量 if/else 或 switch-case。
更易维护、扩展状态逻辑，符合开闭原则。
*/

type State interface {
	PressButton(e *Elevator)
}

type Elevator struct {
	state State
}

func (e *Elevator) SetState(state State) {
	e.state = state
}

func (e *Elevator) PressButton() {
	e.state.PressButton(e)
}

type StopState struct{}

func (s *StopState) PressButton(e *Elevator) {
	fmt.Println("电梯开始运行....")
	e.SetState(&RunningState{})
}

type RunningState struct{}

func (s *RunningState) PressButton(e *Elevator) {
	fmt.Println("电梯正在运行，忽略按钮...")
}

func main() {
	e := &Elevator{}
	e.SetState(&StopState{})
	e.PressButton()
	e.PressButton()
	e.PressButton()
	e.SetState(&StopState{})
	e.PressButton()
}
