package main

import "fmt"

/**
观察者模式（Observer Pattern）是一种行为型设计模式，它定义了一种一对多的依赖关系，
当一个对象（被观察者）状态发生变化时，其所有依赖者（观察者）都会收到通知，并自动更新。
这种模式主要用于构建松耦合的系统，使得对象之间的依赖关系能够在运行时动态建立或解除。
*/

type Observer interface {
	Update(state string)
}

type Subject struct {
	Observers []Observer
	state     string
}

func (s *Subject) Attach(o Observer) {
	s.Observers = append(s.Observers, o)
}

func (s *Subject) Detach(o Observer) {
	for i, observer := range s.Observers {
		if observer == o {
			s.Observers = append(s.Observers[:i], s.Observers[i+1:]...)
			break
		}
	}
}

func (s *Subject) Notify() {
	for _, observer := range s.Observers {
		observer.Update(s.state)
	}
}

func (s *Subject) SetState(state string) {
	s.state = state
	s.Notify()
}

type ConcreteObserver struct {
	name string
}

func (c *ConcreteObserver) Update(state string) {
	fmt.Printf("%s has been updated,state:%s\n", c.name, state)
}

func main() {
	subject := Subject{}
	observer1 := ConcreteObserver{
		name: "Observer1",
	}
	observer2 := ConcreteObserver{
		name: "Observer2",
	}
	subject.Attach(&observer1)
	subject.Attach(&observer2)

	subject.SetState("state1")

	subject.Detach(&observer2)

	subject.SetState("state2")
}
